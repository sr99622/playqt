#ifndef STREAMPANEL_H
#define STREAMPANEL_H

#include <QTextEdit>
#include <QProcess>
#include <QFile>
#include <QRunnable>

#include "Utilities/panel.h"

class Streamer : public QObject, public QRunnable
{
    Q_OBJECT

public:
    Streamer(void *handle, Panel *panel);
    void run() override;

    void *handle;
    Panel *panel;

signals:
    void done();

};

class StreamData : public QByteArray
{

public:
    int position = 0;

};

class StreamPanel : public Panel
{
    Q_OBJECT

public:
    StreamPanel(QMainWindow *parent);

    QTextEdit *text;
    QProcess *process;
    StreamData data;
    //bool first_pass = true;

signals:
    void play();
    //void append(QByteArray*);

public slots:
    void test();
    void clear();
};

#endif // STREAMPANEL_H
