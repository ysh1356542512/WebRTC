package com.dds;

import android.content.Intent;
import android.os.Bundle;
import android.util.Log;
import android.view.View;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import com.dds.nodejs.NodejsActivity;
import com.dds.webrtc.R;
import com.dds.webrtclib.ProxyVideoSink;

import org.webrtc.TFLiteDetector;
import org.webrtc.TFLiteOwner;

public class LauncherActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_launcher);
        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);
        boolean isSuccess = TFLiteOwner.tfLiteDetector.loadModel("SESR240x320_float32.tflite", this);
        Log.d("ysh", "onCreate: "+isSuccess);
    }

    public void nodejs(View view) {
        startActivity(new Intent(this, NodejsActivity.class));
    }

}
