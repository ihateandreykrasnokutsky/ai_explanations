import numpy as np
import time

def deconv_original(x, w, stride=2):
    C_in, H, W = x.shape
    C_in_w, C_out, kH, kW = w.shape
    H_out = kH + (H - 1) * stride
    W_out = kW + (W - 1) * stride
    out = np.zeros((C_out, H_out, W_out))
    
    # Original: cin → cout → i → j
    for cin in range(C_in):
        for cout in range(C_out):
            for i in range(H):
                for j in range(W):
                    out[cout, i*stride:i*stride+kH, j*stride:j*stride+kW] += \
                        x[cin, i, j] * w[cin, cout]
    return out

def deconv_changed(x, w, stride=2):
    C_in, H, W = x.shape
    C_in_w, C_out, kH, kW = w.shape
    H_out = kH + (H - 1) * stride
    W_out = kW + (W - 1) * stride
    out = np.zeros((C_out, H_out, W_out))
    
    # Changed: j → cout → i → cin
    for j in range(W):
        for cout in range(C_out):
            for i in range(H):
                for cin in range(C_in):
                    out[cout, i*stride:i*stride+kH, j*stride:j*stride+kW] += \
                        x[cin, i, j] * w[cin, cout]
    return out

# Benchmark
x = np.random.randn(64, 32, 16).astype(np.float32)
w = np.random.randn(64, 64, 4, 4).astype(np.float32)

# Warm up
_ = deconv_original(x, w)
_ = deconv_changed(x, w)

# Time
start = time.time()
out1 = deconv_original(x, w)
t1 = time.time() - start

start = time.time()
out2 = deconv_changed(x, w)  
t2 = time.time() - start

print(f"Original order: {t1:.3f}s")
print(f"Changed order:  {t2:.3f}s")
print(f"Speed ratio: {t2/t1:.1f}x slower!")
print(f"Outputs equal? {np.allclose(out1, out2, rtol=1e-5)}")

#well, deepseek and google AI say that difference in speed should be 10-1000x, but I don't see any difference in the output. So nevermind then, I'll just continue to study the CNN generoator.

#PS D:\Documents\code>  d:; cd 'd:\Documents\code'; & 'D:\development-resources\Anaconda3\python.exe' 'c:\Users\Andrey Krasnokutsky\.vscode\extensions\ms-python.debugpy-2025.18.0-win32-x64\bundled\libs\debugpy\launcher' '51043' '--' 'D:\Documents\code\ai_explanations\performance difference in shuffled loops.py'
#Original order: 9.732s
#Changed order:  9.783s   
#Speed ratio: 1.0x slower!
#Outputs equal? True