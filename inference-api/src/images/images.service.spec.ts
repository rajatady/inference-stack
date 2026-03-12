import { Test, TestingModule } from '@nestjs/testing';
import { NotFoundException } from '@nestjs/common';
import { of } from 'rxjs';
import { ImagesService } from './images.service';
import { Router } from '../worker-orchestrator/router';
import { MetricsService } from '../metrics/metrics.service';

describe('ImagesService', () => {
  let service: ImagesService;
  let router: Router;
  let mockWorker: any;

  beforeEach(async () => {
    mockWorker = {
      infer: jest.fn(),
    };

    const module: TestingModule = await Test.createTestingModule({
      providers: [
        ImagesService,
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

    service = module.get<ImagesService>(ImagesService);
    router = module.get<Router>(Router);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  it('should generate an image and return base64', async () => {
    const fakeImageBytes = Buffer.from('fake-png-data');

    mockWorker.infer.mockReturnValue(
      of(
        { media: { data: fakeImageBytes, mime_type: 'image/png', is_final: true } },
        { complete: { finish_reason: 'STOP', usage: { prompt_tokens: 3 } } },
      ),
    );

    const result = await service.generate({
      model: 'stabilityai/sd-turbo',
      prompt: 'A cat in space',
    });

    expect(result.created).toBeGreaterThan(0);
    expect(result.data).toHaveLength(1);
    expect(result.data[0].b64_json).toBe(fakeImageBytes.toString('base64'));
    expect(router.route).toHaveBeenCalledWith('stabilityai/sd-turbo');
  });

  it('should throw when worker returns error', async () => {
    mockWorker.infer.mockReturnValue(
      of({
        error: { code: 'INTERNAL', message: 'GPU OOM' },
      }),
    );

    await expect(
      service.generate({ model: 'stabilityai/sd-turbo', prompt: 'test' }),
    ).rejects.toThrow(NotFoundException);
  });

  it('should throw when no media is returned', async () => {
    mockWorker.infer.mockReturnValue(
      of({ complete: { finish_reason: 'STOP', usage: {} } }),
    );

    await expect(
      service.generate({ model: 'stabilityai/sd-turbo', prompt: 'test' }),
    ).rejects.toThrow(NotFoundException);
  });
});
