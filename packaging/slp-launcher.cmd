@echo off
rem Relocatable launcher for the pre-baked slp distribution (Windows).
rem Lives at <dist>\bin\slp.cmd next to a bundled standalone CPython at <dist>\python.
setlocal
set "DIR=%~dp0"
"%DIR%..\python\python.exe" -m commands.cli %*
