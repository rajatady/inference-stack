import { Injectable, Logger, NotFoundException } from '@nestjs/common';
import { lastValueFrom, toArray } from 'rxjs';
import { Router } from '../worker-orchestrator/router';
import { MetricsService } from '../metrics/metrics.service';
import { CreateImageDto } from './dto/create-image.dto';

@Injectable()
export class ImagesService {
  private readonly logger = new Logger(ImagesService.name);

  constructor(
    private readonly router: Router,
    private readonly metricsService: MetricsService,
  ) {}

  async generate(dto: CreateImageDto): Promise<any> {
    const startTime = Date.now();
    const requestId = `img-${Date.now()}`;
    const { worker, workerId } = await this.router.route(dto.model);
    const routingTimeMs = Date.now() - startTime;

    const responses = await lastValueFrom(
      worker
        .infer({
          request_id: requestId,
          model_id: dto.model,
          prompt: dto.prompt,
          params: { max_tokens: 1 },
        })
        .pipe(toArray()),
    );

    // Collect media output
    const mediaResponses = responses.filter((r) => r.media);
    const complete = responses.find((r) => r.complete);
    const error = responses.find((r) => r.error);

    if (error) {
      this.metricsService.recordInference({
        requestId,
        model: dto.model,
        modelType: 'image_gen',
        workerId: workerId ?? '',
        prefillTimeMs: 0,
        decodeTimeMs: 0,
        totalTimeMs: 0,
        queueWaitMs: 0,
        routingTimeMs,
        e2eTimeMs: Date.now() - startTime,
        promptTokens: 0,
        completionTokens: 0,
        cachedTokens: 0,
        userId: 'anonymous',
        priority: 'normal',
        isStream: false,
        finishReason: 'ERROR',
        mediaType: 'image/png',
        isError: true,
        errorMessage: error.error.message,
      });
      throw new NotFoundException(error.error.message);
    }

    if (mediaResponses.length === 0) {
      throw new NotFoundException('No image generated');
    }

    // Combine media data (in case of chunks)
    const allData = Buffer.concat(
      mediaResponses.map((r) => Buffer.from(r.media.data)),
    );

    const usage = complete?.complete?.usage;
    this.metricsService.recordInference({
      requestId,
      model: dto.model,
      modelType: 'image_gen',
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
      mediaType: 'image/png',
      mediaSizeBytes: allData.length,
      isError: false,
    });

    return {
      created: Math.floor(Date.now() / 1000),
      data: [
        {
          b64_json: allData.toString('base64'),
        },
      ],
    };
  }
}
