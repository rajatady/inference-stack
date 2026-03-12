import {
  Controller,
  Get,
  Post,
  Body,
  Param,
  Delete,
  HttpCode,
  HttpException,
  HttpStatus,
  Res,
} from '@nestjs/common';
import { Response } from 'express';
import { v4 as uuidv4 } from 'uuid';
import { CompletionsService } from './completions.service';
import { CreateCompletionDto } from './dto/create-completion.dto';
import { SchedulerService } from '../scheduler/scheduler.service';

@Controller('v1/completions')
export class CompletionsController {
  constructor(
    private readonly completionsService: CompletionsService,
    private readonly scheduler: SchedulerService,
  ) {}

  /**
   * POST /v1/completions
   *
   * OpenAI-compatible completions endpoint.
   * If stream=true, returns SSE. Otherwise returns JSON.
   */
  @Post()
  @HttpCode(200)
  async create(
    @Body() dto: CreateCompletionDto,
    @Res() res: Response,
  ) {
    if (!dto.model) {
      throw new HttpException(
        { error: { message: 'model is required', type: 'invalid_request_error' } },
        HttpStatus.BAD_REQUEST,
      );
    }
    if (!dto.prompt && (!dto.messages || dto.messages.length === 0)) {
      throw new HttpException(
        { error: { message: 'prompt or messages is required', type: 'invalid_request_error' } },
        HttpStatus.BAD_REQUEST,
      );
    }

    // Auto-generate session_id if not provided (for KV cache reuse)
    if (!dto.session_id) {
      dto.session_id = uuidv4();
    }

    if (dto.stream) {
      // SSE streaming response
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.setHeader('X-Accel-Buffering', 'no');

      try {
        const { stream$, cancel } = await this.completionsService.createStream(dto);

        // Cancel inference on client disconnect
        res.on('close', () => {
          cancel();
        });

        stream$.subscribe({
          next: (event) => {
            res.write(`data: ${event.data}\n\n`);
          },
          complete: () => {
            res.end();
          },
          error: (err) => {
            res.write(`data: ${JSON.stringify({ error: { message: err.message } })}\n\n`);
            res.end();
          },
        });
      } catch (err) {
        if (err instanceof HttpException) {
          const status = err.getStatus();
          const body = err.getResponse() as any;
          if (status === 429) {
            res.setHeader('Retry-After', String(body.retryAfter ?? 1));
          }
          res.status(status).json(body);
        } else {
          res.status(500).json({ error: { message: err?.message || 'Internal error' } });
        }
      }
    } else {
      // Non-streaming response
      const { promise, cancel } = this.completionsService.create(dto);

      // Cancel inference on client disconnect
      res.on('close', () => {
        cancel();
      });

      try {
        const result = await promise;
        res.json({ ...result, session_id: dto.session_id });
      } catch (err) {
        if (err instanceof HttpException) {
          const status = err.getStatus();
          const body = err.getResponse() as any;
          if (status === 429) {
            res.setHeader('Retry-After', String(body.retryAfter ?? 1));
          }
          res.status(status).json(body);
        } else {
          const status = err?.error?.code === 'MODEL_NOT_LOADED'
            ? HttpStatus.NOT_FOUND
            : HttpStatus.INTERNAL_SERVER_ERROR;
          res.status(status).json(err);
        }
      }
    }
  }

  /**
   * GET /v1/completions
   * List recent completions (for the UI).
   */
  @Get()
  findAll() {
    return this.completionsService.findAll();
  }

  /**
   * GET /v1/completions/stats
   * Queue and scheduler stats (must be before :id to avoid matching "stats" as an ID).
   */
  @Get('stats')
  getStats() {
    return this.scheduler.getStats();
  }

  /**
   * GET /v1/completions/:id
   * Get a specific completion by ID.
   */
  @Get(':id')
  findOne(@Param('id') id: string) {
    return this.completionsService.findOne(id);
  }

  /**
   * DELETE /v1/completions/:id
   * Delete a completion record.
   */
  @Delete(':id')
  remove(@Param('id') id: string) {
    return this.completionsService.remove(id);
  }
}
