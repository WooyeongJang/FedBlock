package com.fedblock.client.di

import com.fedblock.client.data.api.FedBlockApiService
import com.fedblock.client.data.repository.BlockchainAuditRepositoryImpl
import com.fedblock.client.data.repository.FederatedLearningRepositoryImpl
import com.fedblock.client.domain.repository.BlockchainAuditRepository
import com.fedblock.client.domain.repository.FederatedLearningRepository
import dagger.Binds
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent
import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.util.concurrent.TimeUnit
import javax.inject.Singleton

/**
 * 애플리케이션 모듈
 */
@Module
@InstallIn(SingletonComponent::class)
abstract class AppModule {
    
    @Binds
    @Singleton
    abstract fun bindFederatedLearningRepository(
        impl: FederatedLearningRepositoryImpl
    ): FederatedLearningRepository
    
    @Binds
    @Singleton
    abstract fun bindBlockchainAuditRepository(
        impl: BlockchainAuditRepositoryImpl
    ): BlockchainAuditRepository
    
    companion object {
        
        @Provides
        @Singleton
        fun provideOkHttpClient(): OkHttpClient {
            val loggingInterceptor = HttpLoggingInterceptor().apply {
                level = HttpLoggingInterceptor.Level.BODY
            }
            
            return OkHttpClient.Builder()
                .addInterceptor(loggingInterceptor)
                .connectTimeout(30, TimeUnit.SECONDS)
                .readTimeout(30, TimeUnit.SECONDS)
                .writeTimeout(30, TimeUnit.SECONDS)
                .build()
        }
        
        @Provides
        @Singleton
        fun provideRetrofit(okHttpClient: OkHttpClient): Retrofit {
            return Retrofit.Builder()
                .baseUrl("http://127.0.0.1:5000/") // 기본 서버 URL
                .client(okHttpClient)
                .addConverterFactory(GsonConverterFactory.create())
                .build()
        }
        
        @Provides
        @Singleton
        fun provideFedBlockApiService(retrofit: Retrofit): FedBlockApiService {
            return retrofit.create(FedBlockApiService::class.java)
        }
    }
}
