using System;
using System.Diagnostics;
using System.IO;
using System.Text;

namespace CRYPTAI.Services
{
    public class PythonBackendService
    {
        private Process? _pythonProcess;

        // Logları arayüze göndermek için olay
        public event Action<string>? OnLogReceived;

        // KRİTİK: Onarım gerektiğinde MainWindow'u tetikleyecek olay
        public event Action? OnRecoveryNeeded;

        // Sonsuz döngüye girmemesi için bayrak
        private bool _isRecoveryTriggered = false;

        public void StartBackend(bool forceCheck = false)
        {
            try
            {
                _isRecoveryTriggered = false; // Her başlatmada sıfırla

                string exePath = AppDomain.CurrentDomain.BaseDirectory;
                string projectPath = Path.Combine(exePath, "CryptoBackend");

                if (!Directory.Exists(projectPath))  projectPath = Path.GetFullPath(Path.Combine(exePath, @"..\..\..\CryptoBackend")); 

                if (!Directory.Exists(projectPath))
                {
                    OnLogReceived?.Invoke("KRİTİK HATA: Backend dizini bulunamadı! Operasyon durduruldu.");
                    return;
                }

                // Kütüphane kontrolü (İlk açılış veya zorunlu kontrol)
                if (forceCheck || !Properties.Settings.Default.IsDependenciesInstalled)
                {
                    string reqFile = Path.Combine(projectPath, "requirements.txt");
                    if (File.Exists(reqFile))
                    {
                        OnLogReceived?.Invoke("SİSTEM: Kütüphane bütünlüğü sağlanıyor...");
                        bool installSuccess = InstallDependencies(projectPath);

                        if (installSuccess)
                        {
                            OnLogReceived?.Invoke("SİSTEM: Kütüphaneler doğrulandı.");
                            Properties.Settings.Default.IsDependenciesInstalled = true;
                            Properties.Settings.Default.Save();
                        }
                        else
                        {
                            OnLogReceived?.Invoke("UYARI: Yükleme tam tamamlanamadı, yine de başlatılıyor.");
                        }
                    }
                }
                else
                {
                    OnLogReceived?.Invoke("SİSTEM: Kütüphaneler hazır (Önbellek).");
                }

                // Python sunucusunu başlat
                var psi = new ProcessStartInfo
                {
                    FileName = "cmd.exe",
                    Arguments = "/c python -m uvicorn api_services:app --reload",
                    WorkingDirectory = projectPath,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true, 
                    CreateNoWindow = true,
                    WindowStyle = ProcessWindowStyle.Hidden,
                    StandardOutputEncoding = Encoding.UTF8,
                    StandardErrorEncoding = Encoding.UTF8
                };

                _pythonProcess = new Process();
                _pythonProcess.StartInfo = psi;

                // Normal çıktıları logla
                _pythonProcess.OutputDataReceived += (s, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data)) OnLogReceived?.Invoke(e.Data);
                };

                // Hataları dinle ve analiz et
                _pythonProcess.ErrorDataReceived += (s, e) =>
                {
                    if (!string.IsNullOrEmpty(e.Data))
                    {
                        OnLogReceived?.Invoke($"> {e.Data}");
                        CheckForMissingLibraryError(e.Data);
                    }
                };

                _pythonProcess.Start();
                _pythonProcess.BeginOutputReadLine();
                _pythonProcess.BeginErrorReadLine();

                OnLogReceived?.Invoke("BACKEND: Sunucu başlatılıyor...");
            }
            catch (Exception ex)
            {
                OnLogReceived?.Invoke($"SYSTEM FAILURE: {ex.Message}");
            }
        }

        public void KillProcess()
        {
            try
            {
                if (_pythonProcess is not null && !_pythonProcess.HasExited)
                {
                    Process.Start(new ProcessStartInfo
                    {
                        FileName = "taskkill",
                        Arguments = $"/F /T /PID {_pythonProcess.Id}",
                        CreateNoWindow = true,
                        UseShellExecute = false
                    });
                    _pythonProcess.Dispose();
                }
            }
            catch { }
        }

        // Hata metnini analiz eden metod
        private void CheckForMissingLibraryError(string errorLog)
        {
            if (!_isRecoveryTriggered &&
               (errorLog.Contains("ModuleNotFoundError") ||
                errorLog.Contains("No module named") ||
                errorLog.Contains("ImportError")))
            {
                _isRecoveryTriggered = true;
                OnRecoveryNeeded?.Invoke();
            }
        }

        private bool InstallDependencies(string workingDir)
        {
            try
            {
                var psi = new ProcessStartInfo
                {
                    FileName = "cmd.exe",
                    Arguments = "/c pip install -r requirements.txt --disable-pip-version-check",
                    WorkingDirectory = workingDir,
                    UseShellExecute = false,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    CreateNoWindow = true
                };

                using (var process = new Process())
                {
                    process.StartInfo = psi;
                    process.OutputDataReceived += (s, e) =>
                    {
                        if (!string.IsNullOrWhiteSpace(e.Data) && (e.Data.Contains("Installing") || e.Data.Contains("Successfully")))
                            OnLogReceived?.Invoke($"PIP: {e.Data}");
                    };
                    process.ErrorDataReceived += (s, e) =>
                    {
                        if (!string.IsNullOrWhiteSpace(e.Data)) OnLogReceived?.Invoke($"PIP LOG: {e.Data}");
                    };

                    process.Start();
                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine();

                    bool exited = process.WaitForExit(300000);

                    if (!exited)
                    {
                        process.Kill();
                        return false;
                    }
                    return process.ExitCode == 0;
                }
            }
            catch (Exception ex)
            {
                OnLogReceived?.Invoke($"PIP HATASI: {ex.Message}");
                return false;
            }
        }
    }
}