package com.fedblock.client

import android.app.Application
import dagger.hilt.android.HiltAndroidApp

/**
 * FedBlock Android 클라이언트 애플리케이션
 * 
 * 연합학습과 블록체인 감사 로깅 기능을 제공하는 
 * Android 클라이언트 애플리케이션입니다.
 */
@HiltAndroidApp
class FedBlockApplication : Application() {
    
    override fun onCreate() {
        super.onCreate()
        // 애플리케이션 초기화 로직
    }
}
