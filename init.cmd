@echo off
setlocal EnableExtensions

set "TORCH_BACKEND=cu130"
set "SKIP_SYNC=0"
set "NEW_PROJECT=0"
set "UV_CMD=uv"
set "SYNC_ALL_GROUPS=0"
if not defined UV_CACHE_DIR set "UV_CACHE_DIR=%CD%\.uv-cache"

:parse_args
if "%~1"=="" goto args_done
if /I "%~1"=="--skip-sync" (
    set "SKIP_SYNC=1"
    shift
    goto parse_args
)
if /I "%~1"=="--all-groups" (
    set "SYNC_ALL_GROUPS=1"
    shift
    goto parse_args
)
if /I "%~1"=="cpu" (
    set "TORCH_BACKEND=cpu"
    shift
    goto parse_args
)
if /I "%~1"=="cu126" (
    set "TORCH_BACKEND=cu126"
    shift
    goto parse_args
)
if /I "%~1"=="cu130" (
    set "TORCH_BACKEND=cu130"
    shift
    goto parse_args
)
if /I "%~1"=="cu132" (
    set "TORCH_BACKEND=cu132"
    shift
    goto parse_args
)

echo ERROR: Unknown argument "%~1".
call :usage
exit /b 2

:args_done
echo ========================================
echo   MiniFrontier V1 - Safe Bootstrap
echo ========================================
echo PyTorch backend: %TORCH_BACKEND%
if "%SKIP_SYNC%"=="1" echo Dependency sync: skipped
if "%SYNC_ALL_GROUPS%"=="1" echo Optional dependency groups: enabled
echo.

REM ------------------------------------------------------------
REM 1. Find uv. Install it through Python only when it is absent.
REM ------------------------------------------------------------

where uv >nul 2>&1
if errorlevel 1 (
    where python >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Neither uv nor Python is available in PATH.
        echo Install uv from https://docs.astral.sh/uv/ and retry.
        exit /b 1
    )

    python -m uv --version >nul 2>&1
    if errorlevel 1 (
        echo uv was not found. Attempting a user-level installation...
        python -m pip install --user --upgrade uv
        if errorlevel 1 exit /b 1

        python -m uv --version >nul 2>&1
        if errorlevel 1 (
            echo ERROR: uv installation completed but python -m uv still fails.
            exit /b 1
        )
    )
    set "UV_CMD=python -m uv"
)

%UV_CMD% --version
if errorlevel 1 exit /b 1

REM ------------------------------------------------------------
REM 2. Initialize only a genuinely new project.
REM
REM --bare is intentional: this repository already owns README.md and its
REM instruction files. It prevents uv from creating/replacing template files.
REM ------------------------------------------------------------

if not exist pyproject.toml (
    echo.
    echo Creating the MiniFrontier Python 3.12 package...
    %UV_CMD% init --bare --lib --name minifrontier --python 3.12 .
    if errorlevel 1 exit /b 1

    %UV_CMD% python pin 3.12
    if errorlevel 1 exit /b 1
    set "NEW_PROJECT=1"
) else (
    echo.
    echo Existing pyproject.toml found; project metadata and dependencies are preserved.
    echo The %TORCH_BACKEND% optional backend will be selected for this environment.
    if not exist uv.lock (
        echo No uv.lock found; resuming an incomplete first bootstrap.
        set "NEW_PROJECT=1"
    )
)

REM A packaged uv project must have its import package before the first uv add,
REM because dependency resolution builds the local project as part of syncing.
if not exist "src\minifrontier" mkdir "src\minifrontier"
if not exist "src\minifrontier\__init__.py" type nul > "src\minifrontier\__init__.py"

REM ------------------------------------------------------------
REM 3. Add dependencies only during first initialization.
REM
REM The named PyTorch index pins torch to the requested official wheel source.
REM Other packages continue to resolve from PyPI.
REM ------------------------------------------------------------

if "%NEW_PROJECT%"=="1" (
    echo.
    echo Adding PyTorch from the official %TORCH_BACKEND% wheel index...
    %UV_CMD% add torch --optional "%TORCH_BACKEND%" --index "pytorch-%TORCH_BACKEND%=https://download.pytorch.org/whl/%TORCH_BACKEND%"
    if errorlevel 1 exit /b 1

    echo Adding runtime dependencies...
    %UV_CMD% add numpy "tokenizers>=0.22,<=0.23.0" datasets safetensors jinja2 tqdm
    if errorlevel 1 exit /b 1

    echo Adding development and reference-test dependencies...
    %UV_CMD% add --dev pytest ruff "transformers>=4.57,<6"
    if errorlevel 1 exit /b 1

    echo Adding evaluation and lab dependency groups...
    %UV_CMD% add --group eval lm-eval "chardet<6"
    if errorlevel 1 exit /b 1
    %UV_CMD% add --group labs matplotlib
    if errorlevel 1 exit /b 1
)

REM ------------------------------------------------------------
REM 4. Create directories and package markers without overwriting files.
REM Implementation modules are created by their ordered backlog tasks.
REM ------------------------------------------------------------

echo.
echo Ensuring project directories exist...

for %%D in (
    "src\minifrontier"
    "configs"
    "train"
    "scripts"
    "eval"
    "labs"
    "tests"
    "templates"
    "artifacts"
) do if not exist "%%~D" mkdir "%%~D"

if not exist "src\minifrontier\__init__.py" type nul > "src\minifrontier\__init__.py"

if not exist "artifacts\.gitignore" (
    > "artifacts\.gitignore" echo *
    >> "artifacts\.gitignore" echo !.gitignore
)

if not exist ".gitignore" (
    > ".gitignore" echo .venv/
    >> ".gitignore" echo __pycache__/
    >> ".gitignore" echo .pytest_cache/
    >> ".gitignore" echo .ruff_cache/
    >> ".gitignore" echo .uv-cache/
    >> ".gitignore" echo *.py[cod]
    >> ".gitignore" echo artifacts/*
    >> ".gitignore" echo !artifacts/.gitignore
)

if "%SKIP_SYNC%"=="1" goto complete

REM ------------------------------------------------------------
REM 5. Resolve the locked environment and verify the installation.
REM ------------------------------------------------------------

echo.
echo Synchronizing the project environment...
if "%SYNC_ALL_GROUPS%"=="1" (
    %UV_CMD% sync --extra "%TORCH_BACKEND%" --all-groups
) else (
    %UV_CMD% sync --extra "%TORCH_BACKEND%" --group dev
)
if errorlevel 1 exit /b 1

echo.
echo ========================================
echo   Environment verification
echo ========================================

%UV_CMD% run --no-sync python -c "import sys; print('Python:', sys.version)"
if errorlevel 1 exit /b 1
%UV_CMD% run --no-sync python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('BF16 supported:', torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)"
if errorlevel 1 exit /b 1
%UV_CMD% run --no-sync python -c "import minifrontier; print('MiniFrontier package import: OK')"
if errorlevel 1 exit /b 1

dir /b /s "tests\test_*.py" >nul 2>&1
if errorlevel 1 (
    echo No tests exist yet; pytest is deferred to MF-007 and later tasks.
) else (
    echo Running existing tests...
    %UV_CMD% run --no-sync pytest
    if errorlevel 1 exit /b 1
)

:complete
echo.
echo ========================================
echo   MiniFrontier bootstrap complete
echo ========================================
echo.
if "%NEW_PROJECT%"=="1" (
    echo Next task: MF-002, then MF-003 and MF-004.
) else (
    echo Existing project synchronized without changing source files.
)
echo Read tasks\backlog.md and implement one dependency-ready task at a time.
echo Do not start serious training before the evaluation gate passes.
exit /b 0

:usage
echo Usage: init.cmd [cpu^|cu126^|cu130^|cu132] [--all-groups] [--skip-sync]
echo.
echo Default backend: cu130
echo Use cpu for a lightweight correctness-only environment.
echo Use --all-groups when evaluation and lab dependencies are needed.
exit /b 0
