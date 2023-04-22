package com.dds;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;

import com.dds.webrtc.R;
import com.dds.webrtclib.ProxyVideoSink;

public class PhotoActivity extends AppCompatActivity {
    private static final String TAG = "ysh";
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_photo);

//        ImageView imageView = findViewById(R.id.imageView);
//        imageView.setImageBitmap(StoreBitmap.bitmap);
//        boolean isSuccess = ProxyVideoSink.tfLiteDetector.loadModel("T_SESR240x320_float32.tflite", this);
//        Log.d(TAG, "TFLite load successful?  " + isSuccess);
//        Bitmap newBitmap = ProxyVideoSink.tfLiteDetector.detect3(StoreBitmap.bitmap);
//        if (newBitmap != null) {
//            ImageView imageView2 = findViewById(R.id.imageView2);
//            imageView2.setImageBitmap(StoreBitmap.bitmap2);
//            ImageView imageView3 = findViewById(R.id.imageView3);
//            imageView3.setImageBitmap(newBitmap);
//            Log.d(TAG, "onCreate: newBitmap is not null");
//        } else {
//            Log.d(TAG, "onCreate: newBitmap is null");
//        }

//        Log.d(TAG, "onCreate: " + tfLiteDetector.tfLite.toString());
    }
}