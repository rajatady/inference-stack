import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ServeStaticModule } from '@nestjs/serve-static';
import { join } from 'path';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { CompletionsModule } from './completions/completions.module';
import { ImagesModule } from './images/images.module';
import { AudioModule } from './audio/audio.module';
import { VideoModule } from './video/video.module';
import { Completion } from './completions/entities/completion.entity';
import { MetricsModule } from './metrics/metrics.module';

@Module({
  imports: [
    TypeOrmModule.forRoot({
      type: 'better-sqlite3',
      database: join(__dirname, '..', 'data', 'inference.db'),
      entities: [Completion],
      synchronize: true, // Auto-create tables (dev only)
    }),
    ServeStaticModule.forRoot({
      rootPath: join(__dirname, '..', 'public'),
      serveRoot: '/',
    }),
    MetricsModule,
    CompletionsModule,
    ImagesModule,
    AudioModule,
    VideoModule,
  ],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
