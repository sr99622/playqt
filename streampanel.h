#ifndef STREAMPANEL_H
#define STREAMPANEL_H

#include <QMainWindow>

class StreamPanel : public QWidget
{
    Q_OBJECT

public:
    StreamPanel(QMainWindow *parent);

    QMainWindow *mainWindow;
};

#endif // STREAMPANEL_H
