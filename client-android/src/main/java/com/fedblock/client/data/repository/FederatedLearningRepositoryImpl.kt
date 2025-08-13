package com.fedblock.client.data.repository

import android.content.Context
import com.fedblock.client.data.api.FedBlockApiService
import com.fedblock.client.data.local.ModelParametersDao
import com.fedblock.client.data.ml.MnistModelTrainer
import com.fedblock.client.domain.model.*
import com.fedblock.client.domain.repository.FederatedLearningRepository
import dagger.hilt.android.qualifiers.ApplicationContext
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import javax.inject.Inject
import javax.inject.Singleton

/**
 * 연합학습 저장소 구현체
 */
@Singleton
class FederatedLearningRepositoryImpl @Inject constructor(
    @ApplicationContext private val context: Context,
    private val apiService: FedBlockApiService,
    private val modelParametersDao: ModelParametersDao,
    private val mnistTrainer: MnistModelTrainer
) : FederatedLearningRepository {
    
    private val _clientStatus = MutableStateFlow(ClientStatus.IDLE)
    private val _trainingProgress = MutableStateFlow(
        TrainingProgress(0, 0, 0.0, 0.0, 0L)
    )
    
    override suspend fun registerClient(clientId: String): Result<Boolean> {
        return try {
            val response = apiService.registerClient(clientId)
            if (response.isSuccessful) {
                Result.success(true)
            } else {
                Result.failure(Exception("Registration failed: ${response.message()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    override suspend fun unregisterClient(clientId: String): Result<Boolean> {
        return try {
            val response = apiService.unregisterClient(clientId)
            if (response.isSuccessful) {
                Result.success(true)
            } else {
                Result.failure(Exception("Unregistration failed: ${response.message()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    override suspend fun getGlobalModelParameters(
        trainingType: TrainingType
    ): Result<ModelParameters> {
        return try {
            val response = apiService.getGlobalModelParameters(trainingType.name)
            if (response.isSuccessful) {
                val params = response.body()
                if (params != null) {
                    Result.success(params)
                } else {
                    Result.failure(Exception("Empty response"))
                }
            } else {
                Result.failure(Exception("Failed to get parameters: ${response.message()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    override suspend fun trainLocalModel(
        config: FederatedLearningConfig,
        initialParams: ModelParameters
    ): Result<TrainingResult> {
        return try {
            _clientStatus.value = ClientStatus.TRAINING
            
            val result = when (config.trainingType) {
                TrainingType.MNIST, 
                TrainingType.DETERMINISTIC_MNIST -> {
                    mnistTrainer.trainModel(config, initialParams) { progress ->
                        _trainingProgress.value = progress
                    }
                }
                TrainingType.GOSSIP_MNIST -> {
                    // Gossip 학습 구현
                    mnistTrainer.trainModel(config, initialParams) { progress ->
                        _trainingProgress.value = progress
                    }
                }
                TrainingType.CHEST_X_RAY_PNEUMONIA -> {
                    // 흉부 X-ray 학습 구현 (추후 확장)
                    throw NotImplementedError("Chest X-ray training not implemented yet")
                }
            }
            
            _clientStatus.value = ClientStatus.IDLE
            Result.success(result)
            
        } catch (e: Exception) {
            _clientStatus.value = ClientStatus.ERROR
            Result.failure(e)
        }
    }
    
    override suspend fun uploadModelParameters(
        trainingResult: TrainingResult,
        trainingType: TrainingType
    ): Result<Boolean> {
        return try {
            _clientStatus.value = ClientStatus.UPLOADING
            
            val response = apiService.uploadModelParameters(
                trainingType.name,
                trainingResult
            )
            
            _clientStatus.value = ClientStatus.IDLE
            
            if (response.isSuccessful) {
                Result.success(true)
            } else {
                Result.failure(Exception("Upload failed: ${response.message()}"))
            }
        } catch (e: Exception) {
            _clientStatus.value = ClientStatus.ERROR
            Result.failure(e)
        }
    }
    
    override suspend fun exchangeParametersWithPeers(
        modelParams: ModelParameters,
        peers: List<String>
    ): Result<List<ModelParameters>> {
        return try {
            // 피어들과 모델 파라미터 교환 구현
            val peerParameters = mutableListOf<ModelParameters>()
            
            peers.forEach { peerUrl ->
                try {
                    val response = apiService.getPeerModelParameters(peerUrl)
                    if (response.isSuccessful) {
                        response.body()?.let { params ->
                            peerParameters.add(params)
                        }
                    }
                } catch (e: Exception) {
                    // 개별 피어 실패는 무시하고 계속 진행
                }
            }
            
            Result.success(peerParameters)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }
    
    override fun observeClientStatus(): Flow<ClientStatus> {
        return _clientStatus.asStateFlow()
    }
    
    override fun observeTrainingProgress(): Flow<TrainingProgress> {
        return _trainingProgress.asStateFlow()
    }
}
