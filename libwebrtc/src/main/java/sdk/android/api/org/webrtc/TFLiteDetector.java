package org.webrtc;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.util.Log;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.support.common.ops.CastOp;
import org.tensorflow.lite.support.common.ops.DequantizeOp;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.common.ops.QuantizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;

public class TFLiteDetector {
    public Interpreter tfLite;
    private TensorImage inputImageBuffer;
    private TensorBuffer outputProbabilityBuffer;
    private GpuDelegate gpuDelegate;

    private final String TAG = "ysh";

    private Tensor inputTensor;
    private Tensor outputTensor;

    private static final int INPUT_SIZE_WIDTH = 320;
    private static final int INPUT_SIZE_HEIGHT = 240;
    private static final int CHANNELS = 3;
    private static final int BYTES_PER_CHANNEL = 1;


    /**
     * 加载模型函数
     *
     * @param modelfile
     * @param context
     * @return
     */
    public boolean loadModel(String modelfile, Context context) {
        boolean ret = false;
        Log.d(TAG, "loadModel: start");
        try {
            // 获取在assets中的模型
            MappedByteBuffer modelFile = loadModelFile(context.getAssets(), modelfile);
            // 设置tflite运行条件，使用4线程和GPU进行加速
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4);
//            gpuDelegate = new GpuDelegate();
//            options.addDelegate(gpuDelegate);
            // 实例化tflite
            tfLite = new Interpreter(modelFile, options);
            inputTensor = tfLite.getInputTensor(0);
            outputTensor = tfLite.getOutputTensor(0);

            ret = true;
        } catch (IOException e) {
            e.printStackTrace();
        }
        Log.d(TAG, "loadModel: successful");

        return ret;
    }

    /**
     * 加载模型文件函数
     *
     * @param assets
     * @param modelFilename
     * @return
     * @throws IOException
     */
    private MappedByteBuffer loadModelFile(AssetManager assets, String modelFilename)
            throws IOException {
        Log.d(TAG, "loadModelFile: start");
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = 0;
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.CUPCAKE) {
            declaredLength = fileDescriptor.getDeclaredLength();
        }
        Log.d(TAG, "loadModelFile: successful");
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public Bitmap detect(Bitmap bitmap) {
        DataType imageDataType = tfLite.getInputTensor(0).dataType();
        Log.d(TAG, "imageDataType:" + imageDataType.toString());
        DataType probabilityDataType = tfLite.getOutputTensor(0).dataType();
        Log.d(TAG, "probabilityDataType:" + probabilityDataType.toString());
//        inputTensor.quantizationParams().getScale();
        //定义输入张量图片
        TensorImage yolov5sTfliteInput;
        //定义图片转化器
        ImageProcessor imageProcessor;
        imageProcessor =
                new ImageProcessor.Builder()
                        //转换图片尺寸
                        .add(new ResizeOp(INPUT_SIZE_HEIGHT, INPUT_SIZE_WIDTH, ResizeOp.ResizeMethod.BILINEAR))
                        .add(new QuantizeOp(0, 0.003921568859368563f))
                        //归一化
                        .add(new NormalizeOp(0, 255))
                        //变换参数，zeropoint和scale

                        //最后转化为UINT8
                        .add(new CastOp(DataType.UINT8))
                        .build();
        //new 输入张量图片
        yolov5sTfliteInput = new TensorImage(DataType.UINT8);
        //将我们传入的bitmap加载进去
        yolov5sTfliteInput.load(bitmap);
        Log.d(TAG, "zeroPoint: "+tfLite.getInputTensor(0).quantizationParams().getZeroPoint());
        Log.d(TAG, "Scale: "+tfLite.getInputTensor(0).quantizationParams().getScale());
        Log.d(TAG, "输入图片刚加载："+ Arrays.toString(yolov5sTfliteInput.getBuffer().array()));
        //使用图片转化器将其转化为我们模型需要的尺寸、格式等等
        yolov5sTfliteInput = imageProcessor.process(yolov5sTfliteInput);
        Log.d(TAG, "输入图片预处理后："+Arrays.toString(yolov5sTfliteInput.getBuffer().array()));


        //这里是定义输出接受张量
        int[] outputShape = tfLite.getOutputTensor(0).shape();
        TensorBuffer outputTensorBuffer = TensorBuffer.createFixedSize(outputShape, DataType.UINT8);

        //这是最后转化的图片张量
        TensorImage outputTensorImage = new TensorImage(DataType.UINT8);


        //运行模型
        tfLite.run(yolov5sTfliteInput.getBuffer(), outputTensorBuffer.getBuffer());

        Log.d(TAG, "输出图片刚加载："+ Arrays.toString(outputTensorBuffer.getBuffer().array()));
        //定义输出图片转化器
        ImageProcessor tensorProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(480, 640, ResizeOp.ResizeMethod.BILINEAR))
                .add(new DequantizeOp(5, 0.006305381190031767f))
                .add(new NormalizeOp(0,1/255f))
                .add(new CastOp(DataType.UINT8))
                .build();

        //一系列操作是将输出张量转化为outputTensorImage
//        outputTensorBuffer.loadBuffer(outputTensorBuffer.getBuffer());
        Log.d(TAG, "detect: "+outputTensorBuffer.getShape()[2]);
        Log.d(TAG, "detect: "+outputTensorBuffer.getShape()[1]);
        outputTensorImage.load(outputTensorBuffer);

        //转换成我们需要的
        outputTensorImage = tensorProcessor.process(outputTensorImage);
        Log.d(TAG, "输出图片加载后："+ Arrays.toString(outputTensorImage.getBuffer().array()));
        //得到bitmap
        Bitmap outputBitmap = outputTensorImage.getBitmap();

        return outputBitmap;


    }

    /**
     * 核心函数：接受图片并处理返回
     *
     * @param bitmap
     * @return
     */
    public Bitmap setInputOutputDetails(Bitmap bitmap) {

        int[] inputShape = tfLite.getInputTensor(0).shape();
        DataType inputDataType = tfLite.getInputTensor(0).dataType();
        int[] outputShape = tfLite.getOutputTensor(0).shape();
        DataType outputDataType = tfLite.getOutputTensor(0).dataType();

        // Prepare the input and output buffers.
        ByteBuffer inputBuffer = ByteBuffer.allocateDirect(inputShape[1] *
                inputShape[2] * inputShape[3] * inputDataType.byteSize());
        inputBuffer.order(ByteOrder.nativeOrder());
        inputBuffer.rewind();

        ByteBuffer outputBuffer = ByteBuffer.allocateDirect(outputShape[1] *
                outputShape[2] * outputShape[3] * outputDataType.byteSize());
        outputBuffer.order(ByteOrder.nativeOrder());
        outputBuffer.rewind();

        // Convert the input Bitmap to a ByteBuffer of int8 type.
        ByteBuffer inputTensorBuffer = convertBitmapToByteBuffer(bitmap);

        Log.d(TAG, "run start");

        // Run inference on the input ByteBuffer and get the output ByteBuffer.
        tfLite.run(inputTensorBuffer, outputBuffer);

        Log.d(TAG, "run successful");


        Bitmap outputBitmap = convertByteBufferToBitmap(outputBuffer,
                outputShape[2], outputShape[1]);

        // Process the output Bitmap as desired.
        // ...
        Log.d(TAG, "setInputOutputDetails: " + outputBitmap.toString());

        // Close the interpreter to free up resources.
        tfLite.close();
        return outputBitmap;

        //        ByteBuffer byteBuffer = convertBitmapToByteBuffer(bitmap);
        // 获取模型输入数据格式
//        DataType imageDataType = tfLite.getInputTensor(0).dataType();
//        Log.d(TAG, "imageDataType:" + imageDataType.toString());
//        DataType probabilityDataType = tfLite.getOutputTensor(0).dataType();
//        Log.d(TAG, "probabilityDataType:" + probabilityDataType.toString());
//        if (inputTensor.dataType() == DataType.UINT8) {
//            // 将输入张量的缩放因子和偏移量应用于 ByteBuffer 对象
//            int numBytesPerChannel = inputTensor.numBytes() / inputTensor.shape()[3];
//            for (int i = 0; i < inputTensor.shape()[3]; i++) {
//                for (int j = 0; j < byteBuffer.limit(); j += numBytesPerChannel) {
//                    int pixelValue = (byteBuffer.get(j) & 0xff);
//                    float scaledValue = (pixelValue - INPUT_MEAN) / INPUT_STD;
//                    byteBuffer.putFloat(scaledValue);
//                }
//            }
//        }
//        // 创建TensorImage，用于存放图像数据
//        inputImageBuffer = new TensorImage(imageDataType);
//        inputImageBuffer.load(bitmap);
//
//        // 因为模型的输入shape是任意宽高的图片，即{-1,-1,-1,3}，但是在tflite java版中，我们需要指定输入数据的具体大小。
//        // 所以在这里，我们要根据输入图片的宽高来设置模型的输入的shape
//        int[] inputShape = {1, bitmap.getHeight(), bitmap.getWidth(), 3};
//        tfLite.resizeInput(tfLite.getInputTensor(0).index(), inputShape);
////        Log.e(TAG, "inputShape:" + bitmap.getByteCount());
////        for (int i : inputShape) {
////            Log.e(TAG, i + "");
////        }
//
//        // 获取模型输出数据格式
////        DataType probabilityDataType = tfLite.getOutputTensor(0).dataType();
////        Log.d(TAG, "probabilityDataType:" + probabilityDataType.toString());
//        int scale = 2;
//        // 同样的，要设置模型的输出shape，因为我们用的模型的功能是在原图的基础上，放大scale倍，所以这里要乘以scale
//        int[] probabilityShape = {1, bitmap.getWidth() * scale, bitmap.getHeight() * scale, 3};//tfLite.getOutputTensor(0).shapeSignature();
////        Log.e(TAG, "probabilityShape:");
////        for (int i : probabilityShape) {
////            Log.e(TAG, i + "");
////        }
//
//        // Creates the output tensor and its processor.
//        outputProbabilityBuffer = TensorBuffer.createFixedSize(probabilityShape, probabilityDataType);
    }

    /**
     * Bitmap转byte数组
     *
     * @param bitmap
     * @return
     */
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(INPUT_SIZE_WIDTH * INPUT_SIZE_HEIGHT * CHANNELS * BYTES_PER_CHANNEL);
        byteBuffer.order(ByteOrder.nativeOrder());
        int[] pixels = new int[INPUT_SIZE_WIDTH * INPUT_SIZE_HEIGHT];
        bitmap.getPixels(pixels, 0, INPUT_SIZE_WIDTH, 0, 0, INPUT_SIZE_WIDTH, INPUT_SIZE_HEIGHT);

        int pixel = 0;
        for (int i = 0; i < INPUT_SIZE_HEIGHT; ++i) {
            for (int j = 0; j < INPUT_SIZE_WIDTH; ++j) {
                final int val = pixels[pixel++];
                byteBuffer.put((byte) ((val >> 16) & 0xFF));
                byteBuffer.put((byte) ((val >> 8) & 0xFF));
                byteBuffer.put((byte) (val & 0xFF));
            }
        }
        return byteBuffer;
    }

    /**
     * Byte数组转Bitmap
     *
     * @param byteBuffer
     * @param width
     * @param height
     * @return
     */
    private Bitmap convertByteBufferToBitmap(ByteBuffer byteBuffer, int width, int height) {
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[width * height];
        byteBuffer.rewind();

        int pixel = 0;
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                int r = byteBuffer.get() & 0xFF;
                int g = byteBuffer.get() & 0xFF;
                int b = byteBuffer.get() & 0xFF;
                pixels[pixel++] = 0xFF000000 | (r << 16) | (g << 8) | b;
            }
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
        return bitmap;
    }

    private ByteBuffer normalizeBitmapToFloatBuffer(Bitmap bitmap) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int channels = 3;

        ByteBuffer floatBuffer = ByteBuffer.allocateDirect(width * height * channels*4);
        floatBuffer.order(ByteOrder.nativeOrder());

        int[] pixels = new int[width * height];
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height);

        int pixel = 0;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                final int val = pixels[pixel++];
                float r = ((val >> 16) & 0xFF) / 255.0f;
                float g = ((val >> 8) & 0xFF) / 255.0f;
                float b = (val & 0xFF) / 255.0f;
                floatBuffer.putFloat(r);
                floatBuffer.putFloat(g);
                floatBuffer.putFloat(b);
            }
        }
        floatBuffer.rewind();
        return floatBuffer;
    }

    private ByteBuffer quantizeFloatBufferToByteBuffer(FloatBuffer floatBuffer, float scale, int zeroPoint) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(floatBuffer.capacity());
        Log.d(TAG, "quantizeFloatBufferToByteBuffer: "+floatBuffer.capacity());
        byteBuffer.order(ByteOrder.nativeOrder());
//        floatBuffer.rewind(); // 重置 FloatBuffer 的位置

        while (floatBuffer.hasRemaining()) {
            float floatValue = floatBuffer.get();
            int quantizedValue = Math.round(floatValue / scale) + zeroPoint;
            byteBuffer.put((byte) Math.max(-128, Math.min(127, quantizedValue)));
        }
        byteBuffer.rewind(); // 重置 ByteBuffer 的位置以便读取
        Log.d(TAG, "quantizeFloatBufferToByteBuffer: "+byteBuffer.capacity());
        return byteBuffer;
    }

    private FloatBuffer dequantizeByteBufferToFloatBuffer(ByteBuffer byteBuffer, float scale, int zeroPoint) {
        FloatBuffer floatBuffer = ByteBuffer.allocateDirect(byteBuffer.capacity() * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        byteBuffer.rewind();

        while (byteBuffer.hasRemaining()) {
            int quantizedValue = byteBuffer.get();
            float floatValue = (quantizedValue - zeroPoint) * scale;
            floatBuffer.put(floatValue);
        }
        return floatBuffer;
    }
    private Bitmap convertFloatBufferToBitmap(FloatBuffer floatBuffer, int width, int height) {
        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[width * height];
        floatBuffer.rewind();

        int pixel = 0;
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                float r = floatBuffer.get() * 255.0f;
                float g = floatBuffer.get() * 255.0f;
                float b = floatBuffer.get() * 255.0f;
                int ir = Math.round(Math.min(255.0f, Math.max(0.0f, r)));
                int ig = Math.round(Math.min(255.0f, Math.max(0.0f, g)));
                int ib = Math.round(Math.min(255.0f, Math.max(0.0f, b)));
                int pixelValue = 0xFF000000 | (ir << 16) | (ig << 8) | ib;
                pixels[pixel++] = pixelValue;
            }
        }
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height);
        return bitmap;
    }


    public Bitmap detect2(Bitmap bitmap) {
        float inputScale = tfLite.getInputTensor(0).quantizationParams().getScale();
        int inputZeroPoint = tfLite.getInputTensor(0).quantizationParams().getZeroPoint();
        float outputScale = tfLite.getOutputTensor(0).quantizationParams().getScale();
        int outputZeroPoint = tfLite.getOutputTensor(0).quantizationParams().getZeroPoint();

        ByteBuffer inputFloatBuffer = normalizeBitmapToFloatBuffer(bitmap);
        Log.d(TAG, "detect2: inputFloatBuffer"+ Arrays.toString(inputFloatBuffer.array()));


        ByteBuffer quantizedInputBuffer = quantizeFloatBufferToByteBuffer(inputFloatBuffer.asFloatBuffer(), 0.003921568859368563f, 3);
        Log.d(TAG, "detect2: quantizedInputBuffer"+ Arrays.toString(quantizedInputBuffer.array()));
        int[] outputShape = tfLite.getOutputTensor(0).shape();
        DataType outputDataType = tfLite.getOutputTensor(0).dataType();
        ByteBuffer outputBuffer = ByteBuffer.allocateDirect(outputShape[1] *
                outputShape[2] * outputShape[3] * outputDataType.byteSize());
        outputBuffer.order(ByteOrder.nativeOrder());
        outputBuffer.rewind();
//
//        int[] outputShape = tfLite.getOutputTensor(0).shape();
//        TensorBuffer outputTensorBuffer = TensorBuffer.createFixedSize(outputShape, DataType.UINT8);

        tfLite.run(quantizedInputBuffer, outputBuffer);

        FloatBuffer outputFloatBuffer = dequantizeByteBufferToFloatBuffer(outputBuffer, outputScale, outputZeroPoint);

// 转换回 Bitmap
        int outputWidth = 480;
        int outputHeight = 640;
        Bitmap outputBitmap = convertFloatBufferToBitmap(outputFloatBuffer, outputWidth, outputHeight);

        return outputBitmap;

    }

    public Bitmap detect3(Bitmap bitmap) {

        //进行动态分配输入张量尺寸
//        int inputIndex = 0;
//        int[] newInputDims = {1, bitmap.getWidth(), bitmap.getHeight(), 3};
//        tfLite.resizeInput(inputIndex, newInputDims);
//        tfLite.allocateTensors();



        TensorImage inputTensorImage;
        ImageProcessor inputProcessor;
        ImageProcessor inputProcessor2;
        int width = bitmap.getWidth();
//        Log.d(TAG, "detect3: ");
        int height = bitmap.getHeight();
        Log.d(TAG, "detect3: width:"+width+"height:"+height);

        inputProcessor =
                new ImageProcessor.Builder()
                        .add(new ResizeOp(INPUT_SIZE_HEIGHT, INPUT_SIZE_WIDTH, ResizeOp.ResizeMethod.BILINEAR))
                        .build();
        inputProcessor2 =
                new ImageProcessor.Builder()
                        .add(new QuantizeOp(0,255))
                        .build();
        inputTensorImage = new TensorImage(DataType.FLOAT32);

        inputTensorImage.load(bitmap);
//        Log.d(TAG, "输入图片刚加载："+ Arrays.toString(inputTensorImage.getBuffer().array()));

        inputTensorImage = inputProcessor.process(inputTensorImage);

        inputTensorImage = inputProcessor2.process(inputTensorImage);
//        Log.d(TAG, "输入图片加载后："+ Arrays.toString(inputTensorImage.getBuffer().array()));



        int[] outputShape = tfLite.getOutputTensor(0).shape();
        TensorBuffer outputTensorBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32);

        //这是最后转化的图片张量
        TensorImage outputTensorImage = new TensorImage(DataType.FLOAT32);


        //运行模型
        long start = System.currentTimeMillis();
        tfLite.run(inputTensorImage.getBuffer(), outputTensorBuffer.getBuffer());
        Log.d(TAG, "TFLiteDetector model run "+(System.currentTimeMillis()-start)+"ms");


        //定义输出图片转化器
        ImageProcessor tensorProcessor = new ImageProcessor.Builder()
//
                .add(new NormalizeOp(0,1/255f))
//                .add(new CastOp(DataType.UINT8))
//                .add(new ResizeOp(height, width, ResizeOp.ResizeMethod.BILINEAR))
                .build();

        outputTensorImage.load(outputTensorBuffer);

//        Log.d(TAG, "输出图片刚加载："+ Arrays.toString(outputTensorImage.getBuffer().array()));


        //转换成我们需要的
        outputTensorImage = tensorProcessor.process(outputTensorImage);

//        Log.d(TAG, "输出图片加载后："+ Arrays.toString(outputTensorImage.getBuffer().array()));

        //得到bitmap
        Bitmap outputBitmap = outputTensorImage.getBitmap();
        Log.d(TAG, "detect3: width:"+outputBitmap.getWidth()+"height:"+outputBitmap.getHeight());

        return outputBitmap;


    }

    public Bitmap detect4(ByteBuffer byteBuffer, int width, int height) {





        ImageProcessor inputProcessor;
        TensorBuffer tensorBuffer = byteBufferToTensorBuffer(byteBuffer, width, height);
        TensorImage inputTensorImage = new TensorImage(DataType.FLOAT32);
        inputTensorImage.load(tensorBuffer);
        Log.d(TAG, "detect3: width:" + width + "height:" + height);

        inputProcessor =
                new ImageProcessor.Builder()
//                        .add(new ResizeOp(INPUT_SIZE_HEIGHT, INPUT_SIZE_WIDTH, ResizeOp.ResizeMethod.BILINEAR))
                        .add(new NormalizeOp(0, 255))
                        .build();
        inputTensorImage = new TensorImage(DataType.FLOAT32);

//        Log.d(TAG, "输入图片刚加载："+ Arrays.toString(inputTensorImage.getBuffer().array()));

        inputTensorImage = inputProcessor.process(inputTensorImage);
//        Log.d(TAG, "输入图片加载后："+ Arrays.toString(inputTensorImage.getBuffer().array()));

        int[] outputShape = tfLite.getOutputTensor(0).shape();
        TensorBuffer outputTensorBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32);
        //这是最后转化的图片张量
        TensorImage outputTensorImage = new TensorImage(DataType.FLOAT32);
        //运行模型
        tfLite.run(inputTensorImage.getBuffer(), outputTensorBuffer.getBuffer());

        //定义输出图片转化器
        ImageProcessor tensorProcessor = new ImageProcessor.Builder()
                .add(new NormalizeOp(0, 1 / 255f))
                .build();
        outputTensorImage.load(outputTensorBuffer);
//        Log.d(TAG, "输出图片刚加载："+ Arrays.toString(outputTensorImage.getBuffer().array()));
        //转换成我们需要的
        outputTensorImage = tensorProcessor.process(outputTensorImage);
//        Log.d(TAG, "输出图片加载后："+ Arrays.toString(outputTensorImage.getBuffer().array()));
        //得到bitmap
        Bitmap outputBitmap = outputTensorImage.getBitmap();
        Log.d(TAG, "detect3: width:" + outputBitmap.getWidth() + "height:" + outputBitmap.getHeight());

        return outputBitmap;

    }

    public static TensorBuffer byteBufferToTensorBuffer(ByteBuffer byteBuffer, int width, int height) {
        // Calculate the total number of elements in the buffer
        int numElements = width * height * 3 / 2;

        // Create a new byte array to hold the buffer data
        byte[] byteData = new byte[numElements];

        // Extract the YUV data from the buffer
        byteBuffer.rewind();
        byteBuffer.get(byteData, 0, numElements);

        // Convert the YUV data to RGB and normalize it
        float[] data = new float[numElements];
        for (int i = 0; i < numElements; i += 3) {
            float y = byteData[i] & 0xff;
            float u = byteData[i + 1] & 0xff;
            float v = byteData[i + 2] & 0xff;

            float r = y + 1.13983f * v;
            float g = y - 0.39465f * u - 0.58060f * v;
            float b = y + 2.03211f * u;

            data[i] = r / 255.0f;
            data[i + 1] = g / 255.0f;
            data[i + 2] = b / 255.0f;
        }

        // Create a new TensorBuffer object
        TensorBuffer tensorBuffer = TensorBuffer.createFixedSize(new int[]{1,height, width, 3}, DataType.FLOAT32);

        // Copy the data to the TensorBuffer object
        tensorBuffer.loadArray(data);

        return tensorBuffer;
    }



}