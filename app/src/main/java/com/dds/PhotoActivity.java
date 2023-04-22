package com.dds;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.os.Bundle;
import android.util.Log;
import android.widget.ImageView;

import com.dds.webrtc.R;
import com.dds.webrtclib.ProxyVideoSink;

import org.webrtc.StoreBitmap;

public class PhotoActivity extends AppCompatActivity {
    private static final String TAG = "ysh";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_photo);

        ImageView imageView = findViewById(R.id.imageView);
        imageView.setImageBitmap(StoreBitmap.bitmap);

        ImageView imageView1 = findViewById(R.id.imageView2);
        imageView1.setImageBitmap(StoreBitmap.bitmap2);
    }
}