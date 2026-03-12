import { Module, Global } from '@nestjs/common';
import { ClickHouseService } from './clickhouse.service';
import { MetricsService } from './metrics.service';
import { MetricsController } from './metrics.controller';

@Global()
@Module({
  providers: [ClickHouseService, MetricsService],
  controllers: [MetricsController],
  exports: [MetricsService],
})
export class MetricsModule {}
