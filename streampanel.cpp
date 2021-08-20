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
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.dwFlags |= STARTF_USESTDHANDLES;
    //si.wShowWindow = FALSE;
    //si.dwFlags |= STARTF_USESHOWWINDOW;

    PROCESS_INFORMATION pi;
    ZeroMemory(&pi, sizeof(pi));

    LPWSTR name = (LPWSTR)L"C:\\Users\\sr996\\Projects\\contrib\\bin\\youtube-dl.exe";
    LPWSTR arg = (LPWSTR)L" https://www.youtube.com/watch?v=xkb7EQgIjMA -o -";

    //if (!CreateProcess(name, arg, NULL, NULL, TRUE, CREATE_NEW_CONSOLE, NULL, NULL, &si, &pi)) {
    if (!CreateProcess(name, arg, NULL, NULL, FALSE, NULL, NULL, NULL, &si, &pi)) {
        cout << "CreateProcess failed: " << GetLastError() << endl;
    }
    else {
        CloseHandle(pi.hProcess);
        CloseHandle(pi.hThread);
        /*
        CloseHandle(stdOutRead);
        CloseHandle(stdInWrite);
        CloseHandle(stdErrRead);
        */
    }

    SetStdHandle(STD_ERROR_HANDLE, stdErrWrite);
    SetStdHandle(STD_OUTPUT_HANDLE, stdOutWrite);
    SetStdHandle(STD_INPUT_HANDLE, stdInRead);

    emit play();

    /*
    Streamer *errStreamer = new Streamer(stdErrRead, this);
    QThreadPool::globalInstance()->tryStart(errStreamer);

    Streamer *outStreamer = new Streamer(stdOutRead, this);
    QThreadPool::globalInstance()->tryStart(outStreamer);
    */
}

Streamer::Streamer(void *handle, Panel *panel)
{
    this->handle = handle;
    this->panel = panel;
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
            ((StreamPanel*)panel)->text->append(QString(chBuf).mid(0, dwRead));
        }
        else if (std == stdOutRead) {
            QByteArray array;
            array.resize(dwRead);
            memcpy(array.data(), chBuf, dwRead);
            ((StreamPanel*)panel)->data += array;
            cout << "data size: " << ((StreamPanel*)panel)->data.size() << endl;
            if (bFirstPass && ((StreamPanel*)panel)->data.size() > 1000000) {
                ((StreamPanel*)panel)->emit play();
                bFirstPass = FALSE;
            }
        }
    }
}

void StreamPanel::clear()
{
    text->setText("");
}
