@ECHO OFF

REM -- paths
set SOURCEDIR=.
set BUILDDIR=_build

REM -- Force sphinx-multiversion as the builder
set SPHINXBUILD=sphinx-multiversion

IF "%1" == "html" (
    %SPHINXBUILD% -a -E -W --keep-going %SOURCEDIR% %BUILDDIR%/html
    GOTO end
)

IF "%1" == "" (
    ECHO.Please use `make.bat html`
    GOTO end
)

:end
