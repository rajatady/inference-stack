import { Test, TestingModule } from '@nestjs/testing';
import { NotFoundException } from '@nestjs/common';
import { of } from 'rxjs';
import { AudioService } from './audio.service';
import { Router } from '../worker-orchestrator/router';
import { MetricsService } from '../metrics/metrics.service';

describe('AudioService', () => {
  let service: AudioService;
  let router: Router;
  let mockWorker: any;

  beforeEach(async () => {
    mockWorker = {
      infer: jest.fn(),
    };

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        AudioService,
        {
          provide: Router,
          useValue: {
            route: jest.fn().mockResolvedValue({
              worker: mockWorker,
              workerId: 'worker-0',
              action: 'direct',
            }),
          },
        },
        { provide: MetricsService, useValue: { recordInference: jest.fn() } },
      ],
    }).compile();

    service = module.get<AudioService>(AudioService);
    router = module.get<Router>(Router);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  it('should generate audio and return WAV buffer', async () => {
    const fakeWavBytes = Buffer.from('RIFF-fake-wav');

    mockWorker.infer.mockReturnValue(
      of(
        { media: { data: fakeWavBytes, mime_type: 'audio/wav', is_final: true } },
        { complete: { finish_reason: 'STOP', usage: { prompt_tokens: 2 } } },
      ),
    );

    const result = await service.speech({
      model: 'hexgrad/Kokoro-82M',
      input: 'Hello world',
    });

    expect(Buffer.isBuffer(result)).toBe(true);
    expect(result.toString()).toBe('RIFF-fake-wav');
    expect(router.route).toHaveBeenCalledWith('hexgrad/Kokoro-82M');
  });

  it('should throw when worker returns error', async () => {
    mockWorker.infer.mockReturnValue(
      of({ error: { code: 'INTERNAL', message: 'TTS failed' } }),
    );

    await expect(
      service.speech({ model: 'hexgrad/Kokoro-82M', input: 'test' }),
    ).rejects.toThrow(NotFoundException);
  });

  it('should throw when no media is returned', async () => {
    mockWorker.infer.mockReturnValue(
      of({ complete: { finish_reason: 'STOP', usage: {} } }),
    );

    await expect(
      service.speech({ model: 'hexgrad/Kokoro-82M', input: 'test' }),
    ).rejects.toThrow(NotFoundException);
  });
});
