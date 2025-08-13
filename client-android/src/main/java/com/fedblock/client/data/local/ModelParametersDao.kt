package com.fedblock.client.data.local

import androidx.room.*
import com.fedblock.client.domain.model.ModelParameters

/**
 * 모델 파라미터 DAO
 */
@Dao
interface ModelParametersDao {
    
    @Query("SELECT * FROM model_parameters ORDER BY timestamp DESC")
    suspend fun getAllModelParameters(): List<ModelParametersEntity>
    
    @Query("SELECT * FROM model_parameters WHERE client_id = :clientId ORDER BY timestamp DESC")
    suspend fun getModelParametersByClientId(clientId: String): List<ModelParametersEntity>
    
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertModelParameters(modelParameters: ModelParametersEntity)
    
    @Delete
    suspend fun deleteModelParameters(modelParameters: ModelParametersEntity)
    
    @Query("DELETE FROM model_parameters WHERE client_id = :clientId")
    suspend fun deleteAllByClientId(clientId: String)
}

/**
 * 모델 파라미터 엔티티
 */
@Entity(tableName = "model_parameters")
data class ModelParametersEntity(
    @PrimaryKey val id: String,
    @ColumnInfo(name = "client_id") val clientId: String,
    @ColumnInfo(name = "weights") val weights: String, // JSON string
    @ColumnInfo(name = "bias") val bias: String, // JSON string
    @ColumnInfo(name = "round") val round: Int,
    @ColumnInfo(name = "timestamp") val timestamp: Long
)
