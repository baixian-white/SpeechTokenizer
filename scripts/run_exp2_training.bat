@echo off
setlocal EnableExtensions

set SEED=42
if not "%~1"=="" set SEED=%~1

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set STAMP=%%i
set RUN_ID=exp2_scit_speech_training_%STAMP%_seed%SEED%
set RUN_DIR=output\experiments\%RUN_ID%
set CONFIG_PATH=%RUN_DIR%\configs\scit_speech_base_config.json

echo Preparing Exp2 run: %RUN_ID%
python scripts\prepare_exp2_config.py --run-id %RUN_ID% --seed %SEED% --base-config config\spt_base_cfg.json --exp1-best-config output\experiments\exp1_nas_distill_run1_seed42\artifacts\best_architecture\best_seanet_config.json
if errorlevel 1 exit /b %errorlevel%

echo Starting Exp2 training with config: %CONFIG_PATH%
accelerate launch scripts\train_example.py --config %CONFIG_PATH%
exit /b %errorlevel%
