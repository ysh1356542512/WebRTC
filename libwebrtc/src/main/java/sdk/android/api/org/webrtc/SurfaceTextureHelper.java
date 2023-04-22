/*
 *  Copyright 2015 The WebRTC project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

package org.webrtc;

import android.annotation.TargetApi;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.SurfaceTexture;
import android.opengl.EGL14;
import android.opengl.EGLConfig;
import android.opengl.EGLDisplay;
import android.opengl.EGLSurface;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLUtils;
import android.os.Build;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;


import java.nio.ByteBuffer;
import java.util.concurrent.Callable;

import org.webrtc.VideoFrame.TextureBuffer;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLContext;

/**
 * Helper class for using a SurfaceTexture to create WebRTC VideoFrames. In order to create WebRTC
 * VideoFrames, render onto the SurfaceTexture. The frames will be delivered to the listener. Only
 * one texture frame can be in flight at once, so the frame must be released in order to receive a
 * new frame. Call stopListening() to stop receiveing new frames. Call dispose to release all
 * resources once the texture frame is released.
 */
public class SurfaceTextureHelper {
  private static final String TAG = "SurfaceTextureHelper";

  /**
   * Construct a new SurfaceTextureHelper sharing OpenGL resources with |sharedContext|. A dedicated
   * thread and handler is created for handling the SurfaceTexture. May return null if EGL fails to
   * initialize a pixel buffer surface and make it current. If alignTimestamps is true, the frame
   * timestamps will be aligned to rtc::TimeNanos(). If frame timestamps are aligned to
   * rtc::TimeNanos() there is no need for aligning timestamps again in
   * PeerConnectionFactory.createVideoSource(). This makes the timestamps more accurate and
   * closer to actual creation time.
   */
  public static SurfaceTextureHelper create(final String threadName,
      final EglBase.Context sharedContext, boolean alignTimestamps,
      final YuvConverter yuvConverter) {

    final HandlerThread thread = new HandlerThread(threadName);
    thread.start();
    final Handler handler = new Handler(thread.getLooper());

    // The onFrameAvailable() callback will be executed on the SurfaceTexture ctor thread. See:
    // http://grepcode.com/file/repository.grepcode.com/java/ext/com.google.android/android/5.1.1_r1/android/graphics/SurfaceTexture.java#195.
    // Therefore, in order to control the callback thread on API lvl < 21, the SurfaceTextureHelper
    // is constructed on the |handler| thread.
    return ThreadUtils.invokeAtFrontUninterruptibly(handler, new Callable<SurfaceTextureHelper>() {

      @Override
      public SurfaceTextureHelper call() {
        try {
          return new SurfaceTextureHelper(sharedContext, handler, alignTimestamps, yuvConverter);
        } catch (RuntimeException e) {
          Logging.e(TAG, threadName + " create failure", e);
          return null;
        }
      }
    });
  }

  /**
   * Same as above with alignTimestamps set to false and yuvConverter set to new YuvConverter.
   *
   * @see #create(String, EglBase.Context, boolean, YuvConverter)
   */
  public static SurfaceTextureHelper create(
      final String threadName, final EglBase.Context sharedContext) {
    return create(threadName, sharedContext, /* alignTimestamps= */ false, new YuvConverter());
  }

  /**
   * Same as above with yuvConverter set to new YuvConverter.
   *
   * @see #create(String, EglBase.Context, boolean, YuvConverter)
   */
  public static SurfaceTextureHelper create(
      final String threadName, final EglBase.Context sharedContext, boolean alignTimestamps) {
    return create(threadName, sharedContext, alignTimestamps, new YuvConverter());
  }

  private final Handler handler;
  private final EglBase eglBase;
  private final SurfaceTexture surfaceTexture;
  private final int oesTextureId;
  private final YuvConverter yuvConverter;
    private final TimestampAligner timestampAligner;

  // These variables are only accessed from the |handler| thread.
    private VideoSink listener;
  // The possible states of this class.
  private boolean hasPendingTexture;
  private volatile boolean isTextureInUse;
  private boolean isQuitting;
  private int frameRotation;
  private int textureWidth;
  private int textureHeight;
  // |pendingListener| is set in setListener() and the runnable is posted to the handler thread.
  // setListener() is not allowed to be called again before stopListening(), so this is thread safe.
    private VideoSink pendingListener;

  private static int num = 1;


  private VideoFrameDrawer frameDrawer;
  private GlRectDrawer drawer;
  boolean mirrorHorizontally = true;
  boolean mirrorVertically = true;
  private final Matrix drawMatrix = new Matrix();
  private GlTextureFrameBuffer bitmapTextureFramebuffer ;
  private boolean isInitGPU = false;




  final Runnable setListenerRunnable = new Runnable() {
    @Override
    public void run() {
      Logging.d(TAG, "Setting listener to " + pendingListener);
      listener = pendingListener;
      pendingListener = null;
      // May have a pending frame from the previous capture session - drop it.
      if (hasPendingTexture) {
        // Calling updateTexImage() is neccessary in order to receive new frames.
        updateTexImage();
        hasPendingTexture = false;
      }
    }
  };

  private SurfaceTextureHelper(EglBase.Context sharedContext, Handler handler,
      boolean alignTimestamps, YuvConverter yuvConverter) {
    if (handler.getLooper().getThread() != Thread.currentThread()) {
      throw new IllegalStateException("SurfaceTextureHelper must be created on the handler thread");
    }
    this.handler = handler;
    this.timestampAligner = alignTimestamps ? new TimestampAligner() : null;
    this.yuvConverter = yuvConverter;

    eglBase = EglBase.create(sharedContext, EglBase.CONFIG_PIXEL_BUFFER);
    try {
      // Both these statements have been observed to fail on rare occasions, see BUG=webrtc:5682.
      eglBase.createDummyPbufferSurface();
      eglBase.makeCurrent();
    } catch (RuntimeException e) {
      // Clean up before rethrowing the exception.
      eglBase.release();
      handler.getLooper().quit();
      throw e;
    }

    oesTextureId = GlUtil.generateTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES);
    surfaceTexture = new SurfaceTexture(oesTextureId);
    setOnFrameAvailableListener(surfaceTexture, (SurfaceTexture st) -> {
      hasPendingTexture = true;
      tryDeliverTextureFrame();
    }, handler);
  }

  @TargetApi(21)
  private static void setOnFrameAvailableListener(SurfaceTexture surfaceTexture,
      SurfaceTexture.OnFrameAvailableListener listener, Handler handler) {
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.LOLLIPOP) {
      surfaceTexture.setOnFrameAvailableListener(listener, handler);
    } else {
      // The documentation states that the listener will be called on an arbitrary thread, but in
      // pratice, it is always the thread on which the SurfaceTexture was constructed. There are
      // assertions in place in case this ever changes. For API >= 21, we use the new API to
      // explicitly specify the handler.
      surfaceTexture.setOnFrameAvailableListener(listener);
    }
  }

  /**
   * Start to stream textures to the given |listener|. If you need to change listener, you need to
   * call stopListening() first.
   */
  public void startListening(final VideoSink listener) {
    if (this.listener != null || this.pendingListener != null) {
      throw new IllegalStateException("SurfaceTextureHelper listener has already been set.");
    }
    this.pendingListener = listener;
    handler.post(setListenerRunnable);
  }

  /**
   * Stop listening. The listener set in startListening() is guaranteded to not receive any more
   * onFrame() callbacks after this function returns.
   */
  public void stopListening() {
    Logging.d(TAG, "stopListening()");
    handler.removeCallbacks(setListenerRunnable);
    ThreadUtils.invokeAtFrontUninterruptibly(handler, () -> {
      listener = null;
      pendingListener = null;
    });
  }

  /**
   * Use this function to set the texture size. Note, do not call setDefaultBufferSize() yourself
   * since this class needs to be aware of the texture size.
   */
  public void setTextureSize(int textureWidth, int textureHeight) {
    if (textureWidth <= 0) {
      throw new IllegalArgumentException("Texture width must be positive, but was " + textureWidth);
    }
    if (textureHeight <= 0) {
      throw new IllegalArgumentException(
          "Texture height must be positive, but was " + textureHeight);
    }
    surfaceTexture.setDefaultBufferSize(textureWidth, textureHeight);
    handler.post(() -> {
      this.textureWidth = textureWidth;
      this.textureHeight = textureHeight;
    });
  }

  /** Set the rotation of the delivered frames. */
  public void setFrameRotation(int rotation) {
    handler.post(() -> this.frameRotation = rotation);
  }

  /**
   * Retrieve the underlying SurfaceTexture. The SurfaceTexture should be passed in to a video
   * producer such as a camera or decoder.
   */
  public SurfaceTexture getSurfaceTexture() {
    return surfaceTexture;
  }

  /** Retrieve the handler that calls onFrame(). This handler is valid until dispose() is called. */
  public Handler getHandler() {
    return handler;
  }

  /**
   * This function is called when the texture frame is released. Only one texture frame can be in
   * flight at once, so this function must be called before a new frame is delivered.
   */
  private void returnTextureFrame() {
    handler.post(() -> {
      isTextureInUse = false;
      if (isQuitting) {
        release();
      } else {
        tryDeliverTextureFrame();
      }
    });
  }

  public boolean isTextureInUse() {
    return isTextureInUse;
  }

  /**
   * Call disconnect() to stop receiving frames. OpenGL resources are released and the handler is
   * stopped when the texture frame has been released. You are guaranteed to not receive any more
   * onFrame() after this function returns.
   */
  public void dispose() {
    Logging.d(TAG, "dispose()");
    ThreadUtils.invokeAtFrontUninterruptibly(handler, () -> {
      isQuitting = true;
      if (!isTextureInUse) {
        release();
      }
    });
  }

  /**
   * Posts to the correct thread to convert |textureBuffer| to I420.
   *
   * @deprecated Use toI420() instead.
   */
  @Deprecated
  public VideoFrame.I420Buffer textureToYuv(final TextureBuffer textureBuffer) {
    return textureBuffer.toI420();
  }

  private void updateTexImage() {
    // SurfaceTexture.updateTexImage apparently can compete and deadlock with eglSwapBuffers,
    // as observed on Nexus 5. Therefore, synchronize it with the EGL functions.
    // See https://bugs.chromium.org/p/webrtc/issues/detail?id=5702 for more info.
    synchronized (EglBase.lock) {
      surfaceTexture.updateTexImage();
    }
  }

  private void tryDeliverTextureFrame() {
    if (handler.getLooper().getThread() != Thread.currentThread()) {
      throw new IllegalStateException("Wrong thread.");
    }
    if (isQuitting || !hasPendingTexture || isTextureInUse || listener == null) {
      return;
    }
    isTextureInUse = true;
    hasPendingTexture = false;

    updateTexImage();

    final float[] transformMatrix = new float[16];
    surfaceTexture.getTransformMatrix(transformMatrix);
    long timestampNs = surfaceTexture.getTimestamp();
    if (timestampAligner != null) {
      timestampNs = timestampAligner.translateTimestamp(timestampNs);
    }
    if (textureWidth == 0 || textureHeight == 0) {
      throw new RuntimeException("Texture size has not been set.");
    }
    final VideoFrame.Buffer buffer =
        new TextureBufferImpl(textureWidth, textureHeight, TextureBuffer.Type.OES, oesTextureId,
            RendererCommon.convertMatrixToAndroidGraphicsMatrix(transformMatrix), handler,
            yuvConverter, this ::returnTextureFrame);
//    final VideoFrame frame = new VideoFrame(buffer, frameRotation, timestampNs);
    final VideoFrame frame = new VideoFrame(buffer, 180, timestampNs);

    long start = System.currentTimeMillis();
    Bitmap bitmap = GetBitmapFromVideoFrame(frame);

    if (num == 100) {
      StoreBitmap.bitmap = bitmap;
      StoreBitmap.bitmap2 = TFLiteOwner.tfLiteDetector.detect3(bitmap);
    }
    num++;
    long tf_start = System.currentTimeMillis();
    bitmap = TFLiteOwner.tfLiteDetector.detect3(bitmap);
    Log.d(TAG, "tryDeliverTextureFrame: model run "+(System.currentTimeMillis()-tf_start)+"ms");
    Log.d(TAG, "tryDeliverTextureFrame: bitmap width"+bitmap.getWidth()+"bitmap height"+bitmap.getHeight());

    VideoFrame newFrame = GetVideoFrameFromBitmap(bitmap, timestampNs+System.currentTimeMillis()-start);
    Log.d(TAG, "tryDeliverTextureFrame: waste time" + (System.currentTimeMillis() - start));

    ((VideoSink) listener).onFrame(newFrame);
//    ((VideoSink) listener).onFrame(frame);
    frame.release();
    newFrame.release();
  }

  public VideoFrame GetVideoFrameFromBitmap(Bitmap bitmap,long timestampNs) {
    long start = System.nanoTime();


    int[] textures = new int[1];
    GLES20.glGenTextures(1, textures, 0);
    GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, textures[0]);

    Matrix matrix = new Matrix();
    matrix.preTranslate(0.5f, 0.5f);
    matrix.preScale(1f, -1f);
    matrix.preTranslate(-0.5f, -0.5f);

    TextureBufferImpl buffer = new TextureBufferImpl(textureWidth, textureHeight,
            VideoFrame.TextureBuffer.Type.RGB, textures[0],  matrix,
            handler, yuvConverter, this ::returnTextureFrame);
    GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D,GLES20.GL_TEXTURE_MIN_FILTER,GLES20.GL_NEAREST);
    GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
    GLUtils.texImage2D(GLES20.GL_TEXTURE_2D, 0, bitmap, 0);
//    TextureBufferImpl buffer = new TextureBufferImpl(bitmap.getWidth(), bitmap.getHeight(), VideoFrame.TextureBuffer.Type.RGB, textures[0], new Matrix(), SurfaceTextureHelper.getHandler(), yuvConverter, null);
//    VideoFrame.I420Buffer i420Buf = yuvConverter.convert(buffer);
    long frameTime = System.nanoTime() - start;
    VideoFrame newFrame = new VideoFrame(buffer, 180, timestampNs) ;
    return newFrame;
  }




  public Bitmap GetBitmapFromVideoFrame(VideoFrame frame){
    try {

      if (!isInitGPU)
      {

        isInitGPU = true;
        initializeEGL();
        bitmapTextureFramebuffer =
                new GlTextureFrameBuffer(GLES20.GL_RGBA);
//        bitmapTextureFramebuffer =
//                new GlTextureFrameBuffer(GLES20.GL_RGB);
        frameDrawer = new VideoFrameDrawer();
        drawer = new GlRectDrawer();
      }
      drawMatrix.reset();
      drawMatrix.preTranslate(0.5f, 0.5f);
      drawMatrix.preScale(mirrorHorizontally ? -1f : 1f, mirrorVertically ? -1f : 1f);
      drawMatrix.preScale(1f, -1f);//We want the output to be upside down for Bitmap.
      drawMatrix.preTranslate(-0.5f, -0.5f);

      Log.d(TAG, "GetBitmapFromVideoFrame: frame width:"+frame.getRotatedWidth()+"frame height:"+frame.getRotatedHeight());



      final int scaledWidth = (int) (1 * frame.getRotatedWidth());
      final int scaledHeight = (int) (1 * frame.getRotatedHeight());
//      final int scaledWidth = (int) (1 * frame.getRotatedHeight());
//      final int scaledHeight = (int) (1 * frame.getRotatedWidth());


      bitmapTextureFramebuffer.setSize(scaledWidth, scaledHeight);
      GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, bitmapTextureFramebuffer.getFrameBufferId());
      GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
              GLES20.GL_TEXTURE_2D, bitmapTextureFramebuffer.getTextureId(), 0);

      GLES20.glClearColor(0/* red */, 0/* green */, 0/* blue */, 0/* alpha */);
      GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
      frameDrawer.drawFrame(frame, drawer, drawMatrix, 0/* viewportX */,
              0/* viewportY */, scaledWidth, scaledHeight);

      final ByteBuffer bitmapBuffer = ByteBuffer.allocateDirect(scaledWidth * scaledHeight * 4);
      GLES20.glViewport(0, 0, scaledWidth, scaledHeight);
      GLES20.glReadPixels(
              0, 0, scaledWidth, scaledHeight, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, bitmapBuffer);

      GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
      GlUtil.checkNoGLES2Error("EglRenderer.notifyCallbacks");

      final Bitmap bitmap = Bitmap.createBitmap(scaledWidth, scaledHeight, Bitmap.Config.ARGB_8888);
      bitmap.copyPixelsFromBuffer(bitmapBuffer);
      return bitmap;

    } catch (Exception e) {
      Log.e(TAG, e.toString());
      return null;
    }

  }
  void initializeEGL() {
    if (((EGL10) EGLContext.getEGL()).eglGetCurrentContext().equals(EGL10.EGL_NO_CONTEXT)) {
      // no current context.

      try {
        EGLDisplay dpy = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY);
        int[] vers = new int[2];
        EGL14.eglInitialize(dpy, vers, 0, vers, 1);
        int[] configAttr = {
                EGL14.EGL_COLOR_BUFFER_TYPE, EGL14.EGL_RGB_BUFFER,
                EGL14.EGL_LEVEL, 0,
                EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
                EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
                EGL14.EGL_NONE
        };
        EGLConfig[] configs = new EGLConfig[1];
        int[] numConfig = new int[1];
        EGL14.eglChooseConfig(dpy, configAttr, 0,
                configs, 0, 1, numConfig, 0);
        if (numConfig[0] == 0) {
          // TROUBLE! No config found.
        }
        EGLConfig config = configs[0];
        int[] surfAttr = {
                EGL14.EGL_WIDTH, 64,
                EGL14.EGL_HEIGHT, 64,
                EGL14.EGL_NONE
        };
        EGLSurface surf = EGL14.eglCreatePbufferSurface(dpy, config, surfAttr, 0);

        int[] ctxAttrib = {
                EGL14.EGL_CONTEXT_CLIENT_VERSION, 2,
                EGL14.EGL_NONE
        };
        android.opengl.EGLContext ctx = EGL14.eglCreateContext(dpy, config, EGL14.EGL_NO_CONTEXT, ctxAttrib, 0);
        EGL14.eglMakeCurrent(dpy, surf, surf, ctx);
      }
      catch (Exception ex)
      {
        Log.e(TAG, "Init Collapse::" + ex.toString());
      }
    }
  }



  private void release() {
    if (handler.getLooper().getThread() != Thread.currentThread()) {
      throw new IllegalStateException("Wrong thread.");
    }
    if (isTextureInUse || !isQuitting) {
      throw new IllegalStateException("Unexpected release.");
    }
    yuvConverter.release();
    GLES20.glDeleteTextures(1, new int[] {oesTextureId}, 0);
    surfaceTexture.release();
    eglBase.release();
    handler.getLooper().quit();
    if (timestampAligner != null) {
      timestampAligner.dispose();
    }
  }
}
