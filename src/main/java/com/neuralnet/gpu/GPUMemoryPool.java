package com.neuralnet.gpu;

import org.jcuda.*;
import org.jcuda.runtime.*;
import org.jocl.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Logger;

/**
 * GPU Memory Pool for efficient memory management
 * Reduces allocation overhead and improves performance
 */
public class GPUMemoryPool {
    private static final Logger logger = Logger.getLogger(GPUMemoryPool.class.getName());
    
    // Memory pools for different sizes
    private static final Map<Integer, Queue<Pointer>> cudaMemoryPools = new ConcurrentHashMap<>();
    private static final Map<Integer, Queue<cl_mem>> openclMemoryPools = new ConcurrentHashMap<>();
    
    // Statistics
    private static long totalAllocated = 0;
    private static long totalFreed = 0;
    private static int poolHits = 0;
    private static int poolMisses = 0;
    
    // Configuration
    private static final int MAX_POOL_SIZE = 100; // Maximum buffers per size
    private static final int MIN_BUFFER_SIZE = 1024; // Minimum size for pooling (bytes)
    private static final int MAX_BUFFER_SIZE = 100 * 1024 * 1024; // Maximum size for pooling (100MB)
    
    /**
     * Allocate GPU memory with pooling
     */
    public static Pointer allocateCUDAMemory(int size) {
        if (size < MIN_BUFFER_SIZE || size > MAX_BUFFER_SIZE) {
            // Direct allocation for small or large buffers
            Pointer ptr = new Pointer();
            JCuda.cudaMalloc(ptr, size);
            totalAllocated += size;
            poolMisses++;
            return ptr;
        }
        
        // Try to get from pool
        Queue<Pointer> pool = cudaMemoryPools.get(size);
        if (pool != null && !pool.isEmpty()) {
            Pointer ptr = pool.poll();
            poolHits++;
            return ptr;
        }
        
        // Allocate new memory
        Pointer ptr = new Pointer();
        JCuda.cudaMalloc(ptr, size);
        totalAllocated += size;
        poolMisses++;
        return ptr;
    }
    
    /**
     * Allocate OpenCL memory with pooling
     */
    public static cl_mem allocateOpenCLMemory(int size, int flags) {
        if (size < MIN_BUFFER_SIZE || size > MAX_BUFFER_SIZE) {
            // Direct allocation for small or large buffers
            cl_mem mem = CL.clCreateBuffer(GPUContext.getOpenCLContext(), flags, size, null, null);
            totalAllocated += size;
            poolMisses++;
            return mem;
        }
        
        // Try to get from pool
        Queue<cl_mem> pool = openclMemoryPools.get(size);
        if (pool != null && !pool.isEmpty()) {
            cl_mem mem = pool.poll();
            poolHits++;
            return mem;
        }
        
        // Allocate new memory
        cl_mem mem = CL.clCreateBuffer(GPUContext.getOpenCLContext(), flags, size, null, null);
        totalAllocated += size;
        poolMisses++;
        return mem;
    }
    
    /**
     * Free CUDA memory to pool
     */
    public static void freeCUDAMemory(Pointer ptr, int size) {
        if (size < MIN_BUFFER_SIZE || size > MAX_BUFFER_SIZE) {
            // Direct free for small or large buffers
            JCuda.cudaFree(ptr);
            totalFreed += size;
            return;
        }
        
        // Add to pool
        Queue<Pointer> pool = cudaMemoryPools.computeIfAbsent(size, k -> new LinkedList<>());
        if (pool.size() < MAX_POOL_SIZE) {
            pool.offer(ptr);
        } else {
            // Pool is full, free directly
            JCuda.cudaFree(ptr);
            totalFreed += size;
        }
    }
    
    /**
     * Free OpenCL memory to pool
     */
    public static void freeOpenCLMemory(cl_mem mem, int size) {
        if (size < MIN_BUFFER_SIZE || size > MAX_BUFFER_SIZE) {
            // Direct free for small or large buffers
            CL.clReleaseMemObject(mem);
            totalFreed += size;
            return;
        }
        
        // Add to pool
        Queue<cl_mem> pool = openclMemoryPools.computeIfAbsent(size, k -> new LinkedList<>());
        if (pool.size() < MAX_POOL_SIZE) {
            pool.offer(mem);
        } else {
            // Pool is full, free directly
            CL.clReleaseMemObject(mem);
            totalFreed += size;
        }
    }
    
    /**
     * Get memory pool statistics
     */
    public static MemoryPoolStats getStats() {
        int cudaPoolSize = cudaMemoryPools.values().stream().mapToInt(Queue::size).sum();
        int openclPoolSize = openclMemoryPools.values().stream().mapToInt(Queue::size).sum();
        
        return new MemoryPoolStats(
            totalAllocated,
            totalFreed,
            poolHits,
            poolMisses,
            cudaPoolSize,
            openclPoolSize,
            cudaMemoryPools.size(),
            openclMemoryPools.size()
        );
    }
    
    /**
     * Clear all memory pools
     */
    public static void clearPools() {
        // Clear CUDA pools
        for (Queue<Pointer> pool : cudaMemoryPools.values()) {
            for (Pointer ptr : pool) {
                JCuda.cudaFree(ptr);
            }
            pool.clear();
        }
        cudaMemoryPools.clear();
        
        // Clear OpenCL pools
        for (Queue<cl_mem> pool : openclMemoryPools.values()) {
            for (cl_mem mem : pool) {
                CL.clReleaseMemObject(mem);
            }
            pool.clear();
        }
        openclMemoryPools.clear();
        
        logger.info("GPU memory pools cleared");
    }
    
    /**
     * Optimize memory pools (remove excess buffers)
     */
    public static void optimizePools() {
        // Reduce CUDA pool sizes
        for (Map.Entry<Integer, Queue<Pointer>> entry : cudaMemoryPools.entrySet()) {
            Queue<Pointer> pool = entry.getValue();
            while (pool.size() > MAX_POOL_SIZE / 2) {
                Pointer ptr = pool.poll();
                if (ptr != null) {
                    JCuda.cudaFree(ptr);
                    totalFreed += entry.getKey();
                }
            }
        }
        
        // Reduce OpenCL pool sizes
        for (Map.Entry<Integer, Queue<cl_mem>> entry : openclMemoryPools.entrySet()) {
            Queue<cl_mem> pool = entry.getValue();
            while (pool.size() > MAX_POOL_SIZE / 2) {
                cl_mem mem = pool.poll();
                if (mem != null) {
                    CL.clReleaseMemObject(mem);
                    totalFreed += entry.getKey();
                }
            }
        }
        
        logger.info("GPU memory pools optimized");
    }
    
    /**
     * Memory pool statistics
     */
    public static class MemoryPoolStats {
        private final long totalAllocated;
        private final long totalFreed;
        private final int poolHits;
        private final int poolMisses;
        private final int cudaPoolSize;
        private final int openclPoolSize;
        private final int cudaPoolCount;
        private final int openclPoolCount;
        
        public MemoryPoolStats(long totalAllocated, long totalFreed, int poolHits, int poolMisses,
                             int cudaPoolSize, int openclPoolSize, int cudaPoolCount, int openclPoolCount) {
            this.totalAllocated = totalAllocated;
            this.totalFreed = totalFreed;
            this.poolHits = poolHits;
            this.poolMisses = poolMisses;
            this.cudaPoolSize = cudaPoolSize;
            this.openclPoolSize = openclPoolSize;
            this.cudaPoolCount = cudaPoolCount;
            this.openclPoolCount = openclPoolCount;
        }
        
        public long getTotalAllocated() { return totalAllocated; }
        public long getTotalFreed() { return totalFreed; }
        public int getPoolHits() { return poolHits; }
        public int getPoolMisses() { return poolMisses; }
        public int getCudaPoolSize() { return cudaPoolSize; }
        public int getOpenclPoolSize() { return openclPoolSize; }
        public int getCudaPoolCount() { return cudaPoolCount; }
        public int getOpenclPoolCount() { return openclPoolCount; }
        
        public double getHitRate() {
            int total = poolHits + poolMisses;
            return total > 0 ? (double) poolHits / total : 0.0;
        }
        
        public long getCurrentUsage() {
            return totalAllocated - totalFreed;
        }
        
        @Override
        public String toString() {
            return String.format(
                "MemoryPoolStats{allocated=%d, freed=%d, current=%d, hits=%d, misses=%d, " +
                "hitRate=%.2f%%, cudaPool=%d/%d, openclPool=%d/%d}",
                totalAllocated, totalFreed, getCurrentUsage(), poolHits, poolMisses,
                getHitRate() * 100, cudaPoolSize, cudaPoolCount, openclPoolSize, openclPoolCount
            );
        }
    }
} 