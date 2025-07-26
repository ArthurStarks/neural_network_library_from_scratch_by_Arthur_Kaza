package com.neuralnet.util;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * Performance monitoring utility for tracking execution times and memory usage
 */
public class PerformanceMonitor {
    private static final Map<String, Long> startTimes = new HashMap<>();
    private static final Map<String, Long> totalTimes = new HashMap<>();
    private static final Map<String, Integer> callCounts = new HashMap<>();
    private static final Runtime runtime = Runtime.getRuntime();
    
    private static long initialMemory;
    private static long peakMemory;
    
    static {
        initialMemory = runtime.totalMemory() - runtime.freeMemory();
        peakMemory = initialMemory;
    }

    /**
     * Start timing a named operation
     */
    public static void startTimer(String operation) {
        startTimes.put(operation, System.nanoTime());
    }

    /**
     * Stop timing a named operation and record the result
     */
    public static void stopTimer(String operation) {
        Long startTime = startTimes.get(operation);
        if (startTime != null) {
            long duration = System.nanoTime() - startTime;
            totalTimes.merge(operation, duration, Long::sum);
            callCounts.merge(operation, 1, Integer::sum);
            startTimes.remove(operation);
        }
    }

    /**
     * Get average execution time for an operation in milliseconds
     */
    public static double getAverageTime(String operation) {
        Integer count = callCounts.get(operation);
        Long total = totalTimes.get(operation);
        if (count != null && total != null && count > 0) {
            return TimeUnit.NANOSECONDS.toMillis(total) / (double) count;
        }
        return 0.0;
    }

    /**
     * Get total execution time for an operation in milliseconds
     */
    public static double getTotalTime(String operation) {
        Long total = totalTimes.get(operation);
        return total != null ? TimeUnit.NANOSECONDS.toMillis(total) : 0.0;
    }

    /**
     * Get call count for an operation
     */
    public static int getCallCount(String operation) {
        return callCounts.getOrDefault(operation, 0);
    }

    /**
     * Update peak memory usage
     */
    public static void updateMemoryUsage() {
        long currentMemory = runtime.totalMemory() - runtime.freeMemory();
        peakMemory = Math.max(peakMemory, currentMemory);
    }

    /**
     * Get current memory usage in MB
     */
    public static double getCurrentMemoryMB() {
        return (runtime.totalMemory() - runtime.freeMemory()) / (1024.0 * 1024.0);
    }

    /**
     * Get peak memory usage in MB
     */
    public static double getPeakMemoryMB() {
        return peakMemory / (1024.0 * 1024.0);
    }

    /**
     * Get initial memory usage in MB
     */
    public static double getInitialMemoryMB() {
        return initialMemory / (1024.0 * 1024.0);
    }

    /**
     * Reset all performance counters
     */
    public static void reset() {
        startTimes.clear();
        totalTimes.clear();
        callCounts.clear();
        initialMemory = runtime.totalMemory() - runtime.freeMemory();
        peakMemory = initialMemory;
    }

    /**
     * Print performance summary
     */
    public static void printSummary() {
        System.out.println("=== PERFORMANCE SUMMARY ===");
        System.out.printf("Memory Usage: %.2f MB (Peak: %.2f MB)\n", 
                         getCurrentMemoryMB(), getPeakMemoryMB());
        System.out.println("Operation Times:");
        
        for (String operation : totalTimes.keySet()) {
            double avgTime = getAverageTime(operation);
            double totalTime = getTotalTime(operation);
            int count = getCallCount(operation);
            System.out.printf("  %s: %.3f ms avg (%.3f ms total, %d calls)\n", 
                            operation, avgTime, totalTime, count);
        }
        System.out.println("==========================");
    }

    /**
     * Auto-closeable timer for try-with-resources
     */
    public static class Timer implements AutoCloseable {
        private final String operation;

        public Timer(String operation) {
            this.operation = operation;
            startTimer(operation);
        }

        @Override
        public void close() {
            stopTimer(operation);
        }
    }
} 