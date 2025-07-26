package com.neuralnet.examples;

import com.neuralnet.gpu.GPUOptimizationRunner;

/**
 * GPU Optimization Example
 * Demonstrates how to run comprehensive GPU optimization
 */
public class GPUOptimizationExample {
    
    public static void main(String[] args) {
        System.out.println("🚀 GPU Optimization Example");
        System.out.println("===========================");
        
        try {
            // Run complete GPU optimization suite
            GPUOptimizationRunner.runCompleteOptimization();
            
            System.out.println("\n✅ GPU optimization completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ GPU optimization failed: " + e.getMessage());
            e.printStackTrace();
        } finally {
            // Cleanup GPU resources
            GPUOptimizationRunner.cleanup();
        }
    }
} 