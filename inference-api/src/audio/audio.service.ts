import { Injectable, Logger, NotFoundException } from '@nestjs/common';
import { lastValueFrom, toArray } from 'rxjs';
import { Router } from '../worker-orchestrator/router';
import { MetricsService } from '../metrics/metrics.service';
import { CreateSpeechDto } from './dto/create-speech.dto';

@Injectable()
export class AudioService {
  private readonly logger = new Logger(AudioService.name);

  constructor(
    private readonly router: Router,
    private readonly metricsService: MetricsService,
  ) {}

  async speech(dto: CreateSpeechDto): Promise<Buffer> {
    const startTime = Date.now();
    const requestId = `tts-${Date.now()}`;
    const { worker, workerId } = await this.router.route(dto.model);
    const routingTimeMs = Date.now() - startTime;

    const responses = await lastValueFrom(
      worker
        .infer({
          request_id: requestId,
          model_id: dto.model,
          prompt: dto.input,
          params: { max_tokens: 1 },
        })
        .pipe(toArray()),
    );

    const mediaResponses = responses.filter((r) => r.media);
    const complete = responses.find((r) => r.complete);
    const error = responses.find((r) => r.error);

    if (error) {
      throw new NotFoundException(error.error.message);
    }

    if (mediaResponses.length === 0) {
      throw new NotFoundException('No audio generated');
    }

    const allData = Buffer.concat(
      mediaResponses.map((r) => Buffer.from(r.media.data)),
    );

    const usage = complete?.complete?.usage;
    this.metricsService.recordInference({
      requestId,
      model: dto.model,
      modelType: 'tts',
      workerId: workerId ?? '',
      prefillTimeMs: usage?.prefill_time_ms ?? 0,
      decodeTimeMs: usage?.decode_time_ms ?? 0,
      totalTimeMs: usage?.total_time_ms ?? 0,
      queueWaitMs: 0,
      routingTimeMs,
      e2eTimeMs: Date.now() - startTime,
      promptTokens: usage?.prompt_tokens ?? 0,
      completionTokens: 0,
      cachedTokens: 0,
      userId: 'anonymous',
      priority: 'normal',
      isStream: false,
      finishReason: 'STOP',
      mediaType: 'audio/wav',
      mediaSizeBytes: allData.length,
      isError: false,
    });

    return allData;
  }
}
