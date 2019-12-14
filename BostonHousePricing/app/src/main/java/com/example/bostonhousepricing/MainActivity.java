package com.example.bostonhousepricing;

import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import android.util.Log;
import android.widget.TextView;
import androidx.annotation.NonNull;
import com.google.android.gms.common.logging.Logger;
import com.google.firebase.ml.common.FirebaseMLException;
import com.google.firebase.ml.custom.*;
import org.tensorflow.lite.Interpreter;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import androidx.appcompat.app.AppCompatActivity;
import android.os.Bundle;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

@SuppressWarnings("deprecation")
public class MainActivity extends AppCompatActivity {
    //Interpreter tflite_interpreter;
    //FirebaseModelInterpreter interpreter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        /*
        FirebaseCustomLocalModel localModel = new FirebaseCustomLocalModel.Builder()
                .setAssetFilePath("model.tflite")
                .build();

        try {
            FirebaseModelInterpreterOptions options =
                    new FirebaseModelInterpreterOptions.Builder(localModel).build();
            interpreter = FirebaseModelInterpreter.getInstance(options);
        } catch (FirebaseMLException e) {
            Log.e("hae", "Builder not loaded!");
            e.printStackTrace();
        }
        */

        //tflite_interpreter = new Interpreter(loadModelFromFile());

        setContentView(R.layout.activity_main);
    }

    public MappedByteBuffer loadModelFromFile() {
        MappedByteBuffer mappedByteBuffer = null;
        try {
            AssetFileDescriptor fileDescriptor = this.getAssets().openFd("model.tflite");
            FileInputStream fileInputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
            FileChannel fileChannel = fileInputStream.getChannel();

            /*
            if (fileChannel == null) {
                Log.e("nulla", "fileChannel is null!");
            }
            else if (fileInputStream == null) {
                Log.e("nulla", "fileInputStream is null!");
            }
            else if (fileDescriptor == null) {
                Log.e("nulla", "fileDescriptor is null!");
            }
             */

            mappedByteBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.getStartOffset(), fileDescriptor.getDeclaredLength());
        }
        catch (Exception e) {
            e.printStackTrace();
        }
        if (mappedByteBuffer == null) {
            Log.e("nulla", "bytefuffer is null!");
        }
        return mappedByteBuffer;
    }

    public void predictVal(View view)
    {
        EditText crimField = (EditText) findViewById(R.id.crimField);
        EditText indusField = (EditText) findViewById(R.id.indusField);
        EditText taxField = (EditText) findViewById(R.id.taxField);
        TextView predictionTextView = (TextView) findViewById(R.id.predictionTextView);

        try {
            float[] crimVal = {Float.parseFloat(crimField.getText().toString())};
            float[] indusVal = {Float.parseFloat(indusField.getText().toString())};
            float[] taxVal = {Float.parseFloat(taxField.getText().toString())};

            float[][] inputVals = {crimVal, indusVal, taxVal};
            float[][] prediction = {{0}};

            /*FirebaseModelInputOutputOptions inputOutputOptions =
                    new FirebaseModelInputOutputOptions.Builder()
                            .setInputFormat(0, FirebaseModelDataType.BYTE, new int[]{1, 224, 224, 3})
                            .setOutputFormat(0, FirebaseModelDataType.FLOAT32, new int[]{1, 5})
                            .build();*/

            /*
            StringBuilder stringBuilder = new StringBuilder();
            stringBuilder.append("\n[{\"\'crim\'\":[");
            stringBuilder.append(crimVal[0]);
            stringBuilder.append("],\"\'indus\'\":[");
            stringBuilder.append(indusVal[0]);
            stringBuilder.append("],\"\'tax\'\":[");
            stringBuilder.append(taxVal[0]);
            stringBuilder.append("]}]");
            */

            //tflite_interpreter.run(stringBuilder, prediction);
            float w1 = -0.1136414f;
            float w2 = -0.0166822f;
            float w3 = 0.0572579f;
            float bias = 1.1900340f;

            float pred = w1 * crimVal[0] + w2 * indusVal[0] + w3 * taxVal[0] + bias;

            predictionTextView.setText(Float.toString(pred));
        } catch (Exception ex)
        {
            // do nothing
        }
    }
}
