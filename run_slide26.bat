@echo off
setlocal
if "%~1"=="" goto help
python slide26_demo.py %*
goto end
:help
echo Contoh:
echo   run_slide26.bat val
echo   run_slide26.bat image --source sample.jpg
echo   run_slide26.bat video --source video.mp4
echo   run_slide26.bat webcam
echo   run_slide26.bat export
echo   run_slide26.bat benchmark
:end
endlocal
