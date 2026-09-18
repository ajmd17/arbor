@echo off
rem Builds the viewer for the web into docs\: the page, the wasm, and the texture maps
rem the built-in species draw with. The same as build.sh, for a Windows prompt.
rem GitHub Pages serves docs\ straight from the branch. To try it locally:
rem
rem   crates\arbor-viewer\web\build.bat
rem   python -m http.server 8080 -d docs      (then open http://localhost:8080)
rem
rem Needs the wasm32-unknown-unknown target, and wasm-bindgen-cli at the same version as
rem the wasm-bindgen crate in Cargo.lock (cargo tree -i wasm-bindgen --target
rem wasm32-unknown-unknown -p arbor-viewer says which):
rem
rem   rustup target add wasm32-unknown-unknown
rem   cargo install --locked wasm-bindgen-cli --version ^<that version^>
setlocal
cd /d "%~dp0..\..\.." || exit /b 1

where wasm-bindgen >nul 2>nul
if errorlevel 1 (
    echo wasm-bindgen is not installed; see the top of %~f0 1>&2
    exit /b 1
)

cargo build --release -p arbor-viewer --target wasm32-unknown-unknown || exit /b 1
if not exist docs mkdir docs
wasm-bindgen --target web --no-typescript --out-dir docs --out-name arbor target\wasm32-unknown-unknown\release\arbor-viewer.wasm || exit /b 1
copy /y crates\arbor-viewer\web\index.html docs\ >nul || exit /b 1
rem Served as it is, not run through Jekyll first.
type nul > docs\.nojekyll

rem The page fetches a species' maps from beside it as the species is picked. Only the
rem sets the built-in species name are taken: the rest of the folder is source art.
if not exist docs\assets\textures mkdir docs\assets\textures
for /f tokens^=2^ delims^=^" %%n in ('findstr /c:"texture: " assets\species\*.ron') do (
    for %%m in (albedo normal roughness) do (
        if exist "assets\textures\%%n_%%m.png" xcopy /d /y /q "assets\textures\%%n_%%m.png" docs\assets\textures\ >nul
    )
)

echo built %cd%\docs
endlocal
