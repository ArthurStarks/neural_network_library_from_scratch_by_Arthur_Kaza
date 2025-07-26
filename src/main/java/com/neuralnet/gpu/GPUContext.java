package com.neuralnet.gpu;

import org.jocl.*;
import org.jcuda.*;
import org.jcuda.runtime.*;
import org.jcuda.runtime.JCuda;
import org.jcuda.runtime.cudaDeviceProp;
import org.jcuda.runtime.JCudaRuntime;

import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * GPU Context Manager for OpenCL and CUDA acceleration
 * Provides unified interface for GPU operations
 */
public class GPUContext {
    private static final Logger logger = Logger.getLogger(GPUContext.class.getName());
    
    // OpenCL context
    private static cl_context openCLContext;
    private static cl_command_queue openCLCommandQueue;
    private static cl_device_id openCLDevice;
    
    // CUDA context
    private static boolean cudaAvailable = false;
    private static int cudaDeviceId = 0;
    
    // GPU information
    private static GPUInfo gpuInfo;
    private static boolean initialized = false;
    
    /**
     * Initialize GPU context with automatic device selection
     */
    public static void initialize() {
        if (initialized) {
            return;
        }
        
        logger.info("Initializing GPU acceleration...");
        
        // Try CUDA first (usually faster for neural networks)
        if (initializeCUDA()) {
            logger.info("CUDA initialization successful");
            cudaAvailable = true;
        } else {
            logger.info("CUDA not available, trying OpenCL...");
            if (initializeOpenCL()) {
                logger.info("OpenCL initialization successful");
            } else {
                logger.warning("No GPU acceleration available, falling back to CPU");
            }
        }
        
        initialized = true;
    }
    
    /**
     * Initialize CUDA context
     */
    private static boolean initializeCUDA() {
        try {
            // Initialize CUDA runtime
            JCuda.cudaSetDevice(0);
            
            // Get device properties
            cudaDeviceProp deviceProperties = new cudaDeviceProp();
            JCuda.cudaGetDeviceProperties(deviceProperties, 0);
            
            gpuInfo = new GPUInfo(
                "CUDA",
                deviceProperties.getNameString(),
                deviceProperties.getTotalGlobalMem(),
                deviceProperties.getMultiProcessorCount(),
                deviceProperties.getMaxThreadsPerBlock()
            );
            
            logger.info("CUDA Device: " + gpuInfo.getName());
            logger.info("Global Memory: " + (gpuInfo.getGlobalMemory() / (1024 * 1024)) + " MB");
            logger.info("Compute Units: " + gpuInfo.getComputeUnits());
            
            return true;
        } catch (Exception e) {
            logger.warning("CUDA initialization failed: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Initialize OpenCL context
     */
    private static boolean initializeOpenCL() {
        try {
            // Enable exceptions
            CL.setExceptionsEnabled(true);
            
            // Get platform
            cl_platform_id[] platforms = new cl_platform_id[1];
            cl_int[] numPlatforms = new int[1];
            CL.clGetPlatformIDs(1, platforms, numPlatforms);
            
            if (numPlatforms[0] == 0) {
                logger.warning("No OpenCL platforms found");
                return false;
            }
            
            // Get device
            cl_device_id[] devices = new cl_device_id[1];
            cl_int[] numDevices = new int[1];
            CL.clGetDeviceIDs(platforms[0], CL.CL_DEVICE_TYPE_GPU, 1, devices, numDevices);
            
            if (numDevices[0] == 0) {
                logger.warning("No OpenCL GPU devices found");
                return false;
            }
            
            openCLDevice = devices[0];
            
            // Create context
            openCLContext = CL.clCreateContext(null, 1, new cl_device_id[]{openCLDevice}, null, null, null);
            
            // Create command queue
            openCLCommandQueue = CL.clCreateCommandQueue(openCLContext, openCLDevice, 0, null);
            
            // Get device info
            long[] globalMemSize = new long[1];
            CL.clGetDeviceInfo(openCLDevice, CL.CL_DEVICE_GLOBAL_MEM_SIZE, Sizeof.cl_long, Pointer.to(globalMemSize), null);
            
            long[] computeUnits = new long[1];
            CL.clGetDeviceInfo(openCLDevice, CL.CL_DEVICE_MAX_COMPUTE_UNITS, Sizeof.cl_uint, Pointer.to(computeUnits), null);
            
            byte[] deviceName = new byte[1024];
            CL.clGetDeviceInfo(openCLDevice, CL.CL_DEVICE_NAME, 1024, Pointer.to(deviceName), null);
            String name = new String(deviceName).trim();
            
            gpuInfo = new GPUInfo(
                "OpenCL",
                name,
                globalMemSize[0],
                (int) computeUnits[0],
                256 // Default work group size
            );
            
            logger.info("OpenCL Device: " + gpuInfo.getName());
            logger.info("Global Memory: " + (gpuInfo.getGlobalMemory() / (1024 * 1024)) + " MB");
            logger.info("Compute Units: " + gpuInfo.getComputeUnits());
            
            return true;
        } catch (Exception e) {
            logger.warning("OpenCL initialization failed: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Check if GPU acceleration is available
     */
    public static boolean isGPUAvaliable() {
        return initialized && (cudaAvailable || openCLContext != null);
    }
    
    /**
     * Check if CUDA is available
     */
    public static boolean isCUDAvailable() {
        return initialized && cudaAvailable;
    }
    
    /**
     * Check if OpenCL is available
     */
    public static boolean isOpenCLAvailable() {
        return initialized && openCLContext != null;
    }
    
    /**
     * Get GPU information
     */
    public static GPUInfo getGPUInfo() {
        return gpuInfo;
    }
    
    /**
     * Get CUDA device ID
     */
    public static int getCudaDeviceId() {
        return cudaDeviceId;
    }
    
    /**
     * Get OpenCL context
     */
    public static cl_context getOpenCLContext() {
        return openCLContext;
    }
    
    /**
     * Get OpenCL command queue
     */
    public static cl_command_queue getOpenCLCommandQueue() {
        return openCLCommandQueue;
    }
    
    /**
     * Get OpenCL device
     */
    public static cl_device_id getOpenCLDevice() {
        return openCLDevice;
    }
    
    /**
     * Cleanup GPU resources
     */
    public static void cleanup() {
        if (openCLContext != null) {
            CL.clReleaseContext(openCLContext);
            openCLContext = null;
        }
        if (openCLCommandQueue != null) {
            CL.clReleaseCommandQueue(openCLCommandQueue);
            openCLCommandQueue = null;
        }
        if (openCLDevice != null) {
            CL.clReleaseDevice(openCLDevice);
            openCLDevice = null;
        }
        
        if (cudaAvailable) {
            JCuda.cudaDeviceReset();
        }
        
        initialized = false;
        logger.info("GPU resources cleaned up");
    }
    
    /**
     * GPU Information class
     */
    public static class GPUInfo {
        private final String type;
        private final String name;
        private final long globalMemory;
        private final int computeUnits;
        private final int maxWorkGroupSize;
        
        public GPUInfo(String type, String name, long globalMemory, int computeUnits, int maxWorkGroupSize) {
            this.type = type;
            this.name = name;
            this.globalMemory = globalMemory;
            this.computeUnits = computeUnits;
            this.maxWorkGroupSize = maxWorkGroupSize;
        }
        
        public String getType() { return type; }
        public String getName() { return name; }
        public long getGlobalMemory() { return globalMemory; }
        public int getComputeUnits() { return computeUnits; }
        public int getMaxWorkGroupSize() { return maxWorkGroupSize; }
        
        @Override
        public String toString() {
            return String.format("GPU[%s]: %s, Memory: %d MB, Units: %d", 
                               type, name, globalMemory / (1024 * 1024), computeUnits);
        }
    }
} 