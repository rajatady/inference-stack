import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Observable, Subject } from 'rxjs';
import { v4 as uuidv4 } from 'uuid';
import { Completion } from './entities/completion.entity';
import { CreateCompletionDto } from './dto/create-completion.dto';
import { SchedulerService } from '../scheduler/scheduler.service';
import { TokenizerService } from '../tokenizer/tokenizer.service';
import { MetricsService } from '../metrics/metrics.service';
import { Priority } from '../scheduler/interfaces';
import { getModelType } from '../config/model-roster';

@Injectable()
export class CompletionsService {
  constructor(
    @InjectRepository(Completion)
    private readonly completionRepo: Repository<Completion>,
    private readonly scheduler: SchedulerService,
    private readonly tokenizer: TokenizerService,
    private readonly metricsService: MetricsService,
  ) {}

  /**
   * Non-streaming completion: enqueues via scheduler, waits for result, saves to DB.
   * Returns { promise, cancel } so controller can wire disconnect handling.
   */
  create(dto: CreateCompletionDto): { promise: Promise<any>; cancel: () => void } {
    const requestId = uuidv4();
    const requestStartTime = Date.now();
    const promptForDb = dto.messages?.length
      ? JSON.stringify(dto.messages)
      : dto.prompt || '';
    const entity = this.completionRepo.create({
      id: requestId,
      model: dto.model,
      prompt: promptForDb,
      stream: false,
      status: 'pending',
      max_tokens: dto.max_tokens ?? 50,
      temperature: dto.temperature ?? 1.0,
      top_p: dto.top_p ?? 1.0,
    });
    this.completionRepo.save(entity);

    const priority = this.parsePriority(dto.priority);
    const userId = dto.user || 'anonymous';

    let cancelledExternally = false;
    let schedulerRequestId: string | undefined;

    const promise = this.scheduler
      .enqueue({ dto, userId, priority })
      .then(async (result) => {
        entity.completion_text = result.choices?.[0]?.text ?? '';
        entity.status = 'completed';
        entity.finish_reason = result.choices?.[0]?.finish_reason ?? '';
        entity.prompt_tokens = result.usage?.prompt_tokens ?? 0;
        entity.completion_tokens = result.usage?.completion_tokens ?? 0;
        entity.total_tokens = result.usage?.total_tokens ?? 0;
        entity.prefill_time_ms = result.usage?.prefill_time_ms ?? 0;
        entity.decode_time_ms = result.usage?.decode_time_ms ?? 0;
        entity.total_time_ms = result.usage?.total_time_ms ?? 0;
        entity.worker_id = result.workerId;
        await this.completionRepo.save(entity);

        // Emit to ClickHouse (fire-and-forget)
        this.metricsService.recordInference({
          requestId,
          model: dto.model,
          modelType: getModelType(dto.model) ?? 'text_gen',
          workerId: result.workerId ?? '',
          prefillTimeMs: result.usage?.prefill_time_ms ?? 0,
          decodeTimeMs: result.usage?.decode_time_ms ?? 0,
          totalTimeMs: result.usage?.total_time_ms ?? 0,
          queueWaitMs: result._timing?.queueWaitMs ?? 0,
          routingTimeMs: result._timing?.routingTimeMs ?? 0,
          e2eTimeMs: Date.now() - requestStartTime,
          promptTokens: result.usage?.prompt_tokens ?? 0,
          completionTokens: result.usage?.completion_tokens ?? 0,
          cachedTokens: result.usage?.cached_tokens ?? 0,
          userId: dto.user || 'anonymous',
          priority: dto.priority || 'normal',
          isStream: false,
          finishReason: result.choices?.[0]?.finish_reason ?? '',
          isError: false,
        });

        // Override ID with entity's ID so the API consumer can fetch by it
        return { ...result, id: requestId };
      })
      .catch(async (err) => {
        if (cancelledExternally) {
          entity.status = 'cancelled';
        } else {
          entity.status = 'error';
          entity.error_message = err?.error?.message || err?.message || String(err);
        }
        await this.completionRepo.save(entity);
        throw err;
      });

    const cancel = () => {
      cancelledExternally = true;
      // Cancel via scheduler if we have the request ID
      const ids = this.scheduler.getQueuedRequestIds();
      // The scheduler assigned the ID internally; for now cancel all matching
      this.scheduler.cancel(requestId);
    };

    return { promise, cancel };
  }

  /**
   * Streaming completion: enqueues via scheduler, returns Observable of SSE events.
   * Returns { stream$, cancel } so controller can wire disconnect handling.
   */
  async createStream(
    dto: CreateCompletionDto,
  ): Promise<{ stream$: Observable<MessageEvent>; cancel: () => void }> {
    const requestId = uuidv4();
    const requestStartTime = Date.now();
    const subject = new Subject<MessageEvent>();

    const promptForDb = dto.messages?.length
      ? JSON.stringify(dto.messages)
      : dto.prompt || '';
    const entity = this.completionRepo.create({
      id: requestId,
      model: dto.model,
      prompt: promptForDb,
      stream: true,
      status: 'streaming',
      max_tokens: dto.max_tokens ?? 50,
      temperature: dto.temperature ?? 1.0,
      top_p: dto.top_p ?? 1.0,
    });
    this.completionRepo.save(entity);

    const priority = this.parsePriority(dto.priority);
    const userId = dto.user || 'anonymous';

    // Enqueue — the scheduler handles routing and inference
    // For streaming, we need the raw Observable from the worker,
    // so we enqueue and pipe the result through our subject
    this.scheduler
      .enqueue({ dto, userId, priority })
      .then(async (result) => {
        // Emit final chunk with the complete result
        if (result.choices?.[0]?.text) {
          subject.next({
            data: JSON.stringify({
              id: requestId,
              object: 'text_completion',
              created: Math.floor(Date.now() / 1000),
              model: dto.model,
              choices: [
                {
                  text: result.choices[0].text,
                  index: 0,
                  finish_reason: result.choices[0].finish_reason,
                },
              ],
              usage: result.usage,
            }),
          } as MessageEvent);
        }
        subject.next({ data: '[DONE]' } as MessageEvent);
        subject.complete();

        entity.completion_text = result.choices?.[0]?.text ?? '';
        entity.status = 'completed';
        entity.prefill_time_ms = result.usage?.prefill_time_ms ?? 0;
        entity.decode_time_ms = result.usage?.decode_time_ms ?? 0;
        entity.total_time_ms = result.usage?.total_time_ms ?? 0;
        await this.completionRepo.save(entity);

        // Emit to ClickHouse (fire-and-forget)
        this.metricsService.recordInference({
          requestId,
          model: dto.model,
          modelType: getModelType(dto.model) ?? 'text_gen',
          workerId: result.workerId ?? '',
          prefillTimeMs: result.usage?.prefill_time_ms ?? 0,
          decodeTimeMs: result.usage?.decode_time_ms ?? 0,
          totalTimeMs: result.usage?.total_time_ms ?? 0,
          queueWaitMs: result._timing?.queueWaitMs ?? 0,
          routingTimeMs: result._timing?.routingTimeMs ?? 0,
          e2eTimeMs: Date.now() - requestStartTime,
          promptTokens: result.usage?.prompt_tokens ?? 0,
          completionTokens: result.usage?.completion_tokens ?? 0,
          cachedTokens: result.usage?.cached_tokens ?? 0,
          userId: dto.user || 'anonymous',
          priority: dto.priority || 'normal',
          isStream: true,
          finishReason: result.choices?.[0]?.finish_reason ?? '',
          isError: false,
        });
      })
      .catch((err) => {
        subject.next({
          data: JSON.stringify({
            error: { message: err?.error?.message || 'Inference failed' },
          }),
        } as MessageEvent);
        subject.complete();
      });

    const cancel = () => {
      this.scheduler.cancel(requestId);
      entity.status = 'cancelled';
      this.completionRepo.save(entity);
    };

    return { stream$: subject.asObservable(), cancel };
  }

  async findAll(): Promise<Completion[]> {
    return this.completionRepo.find({
      order: { created_at: 'DESC' },
      take: 100,
    });
  }

  async findOne(id: string): Promise<Completion | null> {
    return this.completionRepo.findOneBy({ id });
  }

  async remove(id: string): Promise<void> {
    await this.completionRepo.delete(id);
  }

  private parsePriority(p?: string): Priority {
    switch (p) {
      case 'high':
        return Priority.HIGH;
      case 'low':
        return Priority.LOW;
      default:
        return Priority.NORMAL;
    }
  }
}
