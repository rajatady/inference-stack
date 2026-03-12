import { Test, TestingModule } from '@nestjs/testing';
import { NotFoundException } from '@nestjs/common';
import { of } from 'rxjs';
import { VideoService } from './video.service';
import { Router } from '../worker-orchestrator/router';
import { MetricsService } from '../metrics/metrics.service';

describe('VideoService', () => {
  let service: VideoService;
  let router: Router;
  let mockWorker: any;

  beforeEach(async () => {
    mockWorker = {
      infer: jest.fn(),
    };

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        VideoService,
        {
          provide: Router,
          useValue: {
            route: jest.fn().mockResolvedValue({
              worker: mockWorker,
              workerId: 'worker-1',
              action: 'direct',
            }),
          },
        },
        { provide: MetricsService, useValue: { recordInference: jest.fn() } },
      ],
    }).compile();

    service = module.get<VideoService>(VideoService);
    router = module.get<Router>(Router);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  it('should generate video and return MP4 buffer', async () => {
    const fakeMp4Bytes = Buffer.from('fake-mp4-data');

    mockWorker.infer.mockReturnValue(
      of(
        { media: { data: fakeMp4Bytes, mime_type: 'video/mp4', is_final: true } },
        { complete: { finish_reason: 'STOP', usage: { prompt_tokens: 4 } } },
      ),
    );

    const result = await service.generate({
      model: 'THUDM/CogVideoX-2b',
      prompt: 'A sunset over ocean',
    });

    expect(Buffer.isBuffer(result)).toBe(true);
    expect(result.toString()).toBe('fake-mp4-data');
    expect(router.route).toHaveBeenCalledWith('THUDM/CogVideoX-2b');
  });

  it('should throw when worker returns error', async () => {
    mockWorker.infer.mockReturnValue(
      of({ error: { code: 'OOM', message: 'GPU OOM' } }),
    );

    await expect(
      service.generate({ model: 'THUDM/CogVideoX-2b', prompt: 'test' }),
    ).rejects.toThrow(NotFoundException);
  });

  it('should throw when no media is returned', async () => {
    mockWorker.infer.mockReturnValue(
      of({ complete: { finish_reason: 'STOP', usage: {} } }),
    );

    await expect(
      service.generate({ model: 'THUDM/CogVideoX-2b', prompt: 'test' }),
    ).rejects.toThrow(NotFoundException);
  });
});
