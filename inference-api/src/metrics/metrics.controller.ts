import { Controller, Get, Query } from '@nestjs/common';
import { MetricsService } from './metrics.service';

@Controller('v1/metrics')
export class MetricsController {
  constructor(private readonly metricsService: MetricsService) {}

  @Get('tps')
  getTps(@Query('window') window?: string) {
    return this.metricsService.getTps(window || '5m');
  }

  @Get('latency')
  getLatency(@Query('window') window?: string) {
    return this.metricsService.getLatencyPercentiles(window || '5m');
  }

  @Get('breakdown')
  getBreakdown(@Query('window') window?: string) {
    return this.metricsService.getBreakdown(window || '5m');
  }

  @Get('history')
  getHistory(
    @Query('model') model?: string,
    @Query('metric') metric?: string,
    @Query('interval') interval?: string,
    @Query('from') from?: string,
    @Query('to') to?: string,
  ) {
    return this.metricsService.getHistory({
      model,
      metric,
      interval: interval || '1m',
      from,
      to,
    });
  }
}
