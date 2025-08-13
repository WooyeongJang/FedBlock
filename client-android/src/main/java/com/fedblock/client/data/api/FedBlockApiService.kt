package com.fedblock.client.data.api

import com.fedblock.client.domain.model.ModelParameters
import com.fedblock.client.domain.model.TrainingResult
import retrofit2.Response
import retrofit2.http.*

/**
 * FedBlock 서버 API 인터페이스
 */
interface FedBlockApiService {
    
    /**
     * 클라이언트 등록
     */
    @POST("client")
    @FormUrlEncoded
    suspend fun registerClient(
        @Field("client_id") clientId: String
    ): Response<Unit>
    
    /**
     * 클라이언트 해제
     */
    @DELETE("client")
    @FormUrlEncoded
    suspend fun unregisterClient(
        @Field("client_id") clientId: String
    ): Response<Unit>
    
    /**
     * 글로벌 모델 파라미터 가져오기
     */
    @GET("model_params")
    suspend fun getGlobalModelParameters(
        @Query("training_type") trainingType: String
    ): Response<ModelParameters>
    
    /**
     * 훈련된 모델 파라미터 업로드
     */
    @PUT("model_params")
    suspend fun uploadModelParameters(
        @Query("training_type") trainingType: String,
        @Body trainingResult: TrainingResult
    ): Response<Unit>
    
    /**
     * 훈련 시작 요청
     */
    @POST("training")
    suspend fun startTraining(
        @Body trainingRequest: TrainingRequest
    ): Response<Unit>
    
    /**
     * 피어 클라이언트에서 모델 파라미터 가져오기
     */
    @GET
    suspend fun getPeerModelParameters(
        @Url peerUrl: String
    ): Response<ModelParameters>
    
    /**
     * 훈련 라운드 완료 신호
     */
    @POST("finish_round")
    suspend fun finishRound(
        @Body finishRequest: FinishRoundRequest
    ): Response<Unit>
    
    /**
     * 서버 상태 확인
     */
    @GET("health")
    suspend fun checkServerHealth(): Response<ServerHealthResponse>
}

/**
 * 훈련 요청 데이터
 */
data class TrainingRequest(
    val trainingType: String,
    val learningRate: Double,
    val epochs: Int,
    val batchSize: Int,
    val clientId: String,
    val round: Int,
    val roundSize: Int,
    val clients: List<PeerClient>? = null
)

/**
 * 피어 클라이언트 정보
 */
data class PeerClient(
    val clientId: String,
    val clientUrl: String
)

/**
 * 라운드 완료 요청
 */
data class FinishRoundRequest(
    val clientId: String,
    val trainingType: String
)

/**
 * 서버 상태 응답
 */
data class ServerHealthResponse(
    val status: String,
    val timestamp: Long,
    val activeClients: Int
)
