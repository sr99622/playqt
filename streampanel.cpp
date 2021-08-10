#include "streampanel.h"
#include <QPushButton>
#include <QGridLayout>
#include <QThreadPool>
#include <iostream>
#include <tchar.h>
#include <windows.h>

using namespace std;

#define BUFSIZE 8192

HANDLE stdInRead;
HANDLE stdInWrite;
HANDLE stdOutRead;
HANDLE stdOutWrite;
HANDLE stdErrRead;
HANDLE stdErrWrite;

StreamPanel::StreamPanel(QMainWindow *parent) : Panel(parent)
{
    text = new QTextEdit();
    QPushButton *test = new QPushButton("Test");
    test->setMaximumWidth(60);
    connect(test, SIGNAL(clicked()), this, SLOT(test()));
    QPushButton *clear = new QPushButton("Clear");
    clear->setMaximumWidth(60);
    connect(clear, SIGNAL(clicked()), this, SLOT(clear()));

    QGridLayout *layout = new QGridLayout();
    layout->addWidget(text,    0, 0, 1, 4);
    layout->addWidget(test,    1, 2, 1, 1);
    layout->addWidget(clear,   1, 3, 1, 1);
    setLayout(layout);
}

void StreamPanel::test()
{
    SECURITY_ATTRIBUTES sa;
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    sa.lpSecurityDescriptor = NULL;

    if (!CreatePipe(&stdOutRead, &stdOutWrite, &sa, 0)) {
        cout << "stdOut create pipe error: " << GetLastError() << endl;
    }

    if (!SetHandleInformation(stdOutRead, HANDLE_FLAG_INHERIT, 0)) {
        cout << "stdOut set handle information: " << GetLastError() << endl;
    }

    if (!CreatePipe(&stdInRead, &stdInWrite, &sa, 0)) {
        cout << "stdIn create pipe error: " << GetLastError() << endl;
    }

    if (!SetHandleInformation(stdInWrite, HANDLE_FLAG_INHERIT, 0)) {
        cout << "stdIn set handle information: " << GetLastError() << endl;
    }

    if (!CreatePipe(&stdErrRead, &stdErrWrite, &sa, 0)) {
        cout << "StdErr create pipe error: " << GetLastError() << endl;
    }

    if (!SetHandleInformation(stdErrRead, HANDLE_FLAG_INHERIT, 0)) {
        cout << "stdErr set handle information: " << GetLastError() << endl;
    }

    STARTUPINFO si;
    ZeroMemory(&si, sizeof(si));
    si.cb = sizeof(si);
    si.hStdError = stdErrWrite;
    si.hStdOutput = stdOutWrite;
    si.hStdInput = stdInRead;
    si.dwFlags |= STARTF_USESTDHANDLES;
    /*
    si.dwX = 100;
    si.dwY = 100;
    si.dwFlags |= STARTF_USEPOSITION;
    si.dwXSize = 100;
    si.dwYSize = 100;
    si.dwFlags |= STARTF_USESIZE;
    */
    si.wShowWindow = FALSE;
    si.dwFlags |= STARTF_USESHOWWINDOW;

    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(pi));

    LPWSTR name = (LPWSTR)L"C:\\Users\\sr996\\Projects\\contrib\\bin\\youtube-dl.exe";
    LPWSTR arg = (LPWSTR)L" https://www.youtube.com/watch?v=UaxBoj_4ISk -o -";

    if (!CreateProcess(name, arg, NULL, NULL, TRUE, CREATE_NEW_CONSOLE | CREATE_NO_WINDOW, NULL, NULL, &si, &pi)) {
        cout << "CreateProcess failed: " << GetLastError() << endl;
    }
    else {

        /*
        cout << "CreateProcess success: " << endl;
        CONSOLE_SCREEN_BUFFER_INFO info;
        BOOL result = GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info);
        if (result) {
            cout << "GetConsoleScreenBufferInfo success" << endl;
        }
        else {
            cout << "GetConsoleScreenBufferInfo fail" << endl;
        }
        */

        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        CloseHandle(stdOutWrite);
        CloseHandle(stdInRead);
        CloseHandle(stdErrWrite);
    }

    //writeToPipe();

    Streamer *errStreamer = new Streamer(stdErrRead, this);
    QThreadPool::globalInstance()->tryStart(errStreamer);

    Streamer *outStreamer = new Streamer(stdOutRead, this);
    QThreadPool::globalInstance()->tryStart(outStreamer);


    //readFromPipe();
    //readFromErr();


    cout << "test success" << endl;
}

void StreamPanel::readFromPipe()
{
    DWORD dwRead;
    BOOL bSuccess = FALSE;
    CHAR chBuf[BUFSIZE];
    for (;;)
    {
        bSuccess = ReadFile(stdOutRead, chBuf, BUFSIZE, &dwRead, NULL);
        if (!bSuccess || dwRead == 0)
            break;

        QByteArray array;
        array.resize(dwRead);
        memcpy(array.data(), chBuf, dwRead);

        cout << "ReadFile: " << dwRead << " data size: " << data.size() << endl;

        data += array;

    }

    emit play();
}

void StreamPanel::readFromErr()
{
    cout << "start readFromErr" << endl;
    DWORD dwRead;
    CHAR chBuf[BUFSIZE];
    BOOL bSuccess = FALSE;
    for (;;)
    {
        bSuccess = ReadFile(stdErrRead, chBuf, BUFSIZE, &dwRead, NULL);
        if (!bSuccess || dwRead == 0)
            break;

        cout << "ReadFile: " << dwRead << endl;

        text->append(QString(chBuf).mid(0, dwRead));
    }
}

Streamer::Streamer(void *handle, Panel *panel)
{
    this->handle = handle;
    this->panel = panel;
    //setAutoDelete(false);
}

void Streamer::run()
{
    HANDLE std = (HANDLE)handle;
    DWORD dwRead;
    CHAR chBuf[BUFSIZE];
    BOOL bSuccess = FALSE;
    BOOL bFirstPass = TRUE;

    for (;;) {
        bSuccess = ReadFile(std, chBuf, BUFSIZE, &dwRead, NULL);
        if (!bSuccess || dwRead == 0)
            break;

        if (std == stdErrRead) {
            cout << "errStreamer::run: " << dwRead << endl;
            ((StreamPanel*)panel)->text->append(QString(chBuf).mid(0, dwRead));
        }
        else if (std == stdOutRead) {
            QByteArray array;
            array.resize(dwRead);
            memcpy(array.data(), chBuf, dwRead);
            ((StreamPanel*)panel)->data += array;
            cout << "stdStreamer: " << ((StreamPanel*)panel)->data.size() << endl;
            if (bFirstPass && ((StreamPanel*)panel)->data.size() > 1000000) {
                cout << "START PLAYER--------------------------------------------------------" << endl;
                ((StreamPanel*)panel)->emit play();
                bFirstPass = FALSE;
            }
        }
    }
    cout << "Steamer done: " << endl;
    //if (std == stdOutRead)
    //    ((StreamPanel*)panel)->emit play();
}

void StreamPanel::keeper()
{
    cout << "StreamPanel::test" << endl;
    data.clear();
    data.position = 0;

    QStringList arguments;
    arguments << "https://www.youtube.com/watch?v=UaxBoj_4ISk";
    arguments << "-o";
    arguments << "-";
    process = new QProcess(this);
    connect(process, SIGNAL(readyReadStandardError()), this, SLOT(readStdErr()));
    connect(process, SIGNAL(readyReadStandardOutput()), this, SLOT(readStdOut()));
    connect(process, SIGNAL(finished(int, QProcess::ExitStatus)), this, SLOT(finished(int, QProcess::ExitStatus)));
    cout << "testing 1, 2, 3" << endl;
    //process->startDetached("youtube-dl", arguments);
    process->setProgram("youtube-dl");
    process->setArguments(arguments);
    process->setStandardOutputFile("tiger.mp4");
    process->startDetached();
    //MW->control()->play();
}

void StreamPanel::clear()
{
    text->setText("");
}

void StreamPanel::readStdOut()
{
    cout << "available bytes: " << process->bytesAvailable() << endl;
    QByteArray array = process->readAllStandardOutput();
    data += array;
}

void StreamPanel::readStdErr()
{
    QByteArray array = process->readAllStandardError();
    text->append((QString)array);
}

void StreamPanel::finished(int exitCode, QProcess::ExitStatus exitStatus)
{
    text->append(QString("exitCode: %1").arg(exitCode));

    switch (exitStatus) {
    case QProcess::NormalExit:
        text->append("Normal Exit");
        break;
    case QProcess::CrashExit:
        text->append("Crash Exit");
        break;
    }
    //MW->control()->play();
}

void StreamPanel::writeToPipe()
{
    /*
    cout << "writing to pipe" << endl;
    DWORD dwRead, dwWritten;
    CHAR chBuf[256];
    BOOL bSuccess = FALSE;

    strcpy(chBuf, "-h\r\n");
    dwRead = 4;
    bSuccess = WriteFile(stdInWrite, chBuf, dwRead, &dwWritten, NULL);
    if (!bSuccess) {
        cout << "write failure: " << GetLastError() << endl;
    }
    else {
        cout << "wrote " << dwWritten << " characters to child process" << endl;
    }
    */
}



//bSuccess = CreateProcess(0, szBuffer, 0, 0, FALSE, CREATE_NEW_CONSOLE, 0, 0, &structStartupInfo, &structProcInfo);
//    if (!CreateProcess(name, NULL, NULL, NULL, FALSE, 0x00010000, NULL, NULL, &si, &pi)) {

