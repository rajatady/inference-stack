import { Subscription } from 'rxjs';
import { CreateCompletionDto } from '../completions/dto/create-completion.dto';

export enum Priority {
  HIGH = 0,
  NORMAL = 1,
  LOW = 2,
}

export interface SchedulerConfig {
  maxQueueDepth: number;
  maxQueuedTokens: number;
  agingBoostPerSecond: number;
  /** Per-request timeout in ms. 0 = no timeout. Default: 60s. */
  requestTimeoutMs: number;
}

export const DEFAULT_SCHEDULER_CONFIG: SchedulerConfig = {
  maxQueueDepth: 100,
  maxQueuedTokens: 50_000,
  agingBoostPerSecond: 0.1,
  requestTimeoutMs: 120_000,
};

export type RequestState =
  | 'queued'
  | 'routing'
  | 'active'
  | 'completed'
  | 'error'
  | 'cancelled'
  | 'timeout';

export interface QueuedRequest {
  id: string;
  dto: CreateCompletionDto;
  userId: string;
  priority: Priority;
  estimatedTokens: number;
  enqueuedAt: number;
  effectivePriority: number;
  state: RequestState;
  resolve: (value: any) => void;
  reject: (reason: any) => void;
  subscription?: Subscription;
  /** Timestamp when dispatchRequest() starts (for gateway timing) */
  dispatchStartTime?: number;
  /** Time spent routing (dispatch start → worker.infer() called) */
  routingTimeMs?: number;
  /** Timer handle for per-request timeout */
  timeoutTimer?: ReturnType<typeof setTimeout>;
}
