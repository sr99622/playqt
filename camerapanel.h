#ifndef CAMERAPANEL_H
#define CAMERAPANEL_H

#include <QMainWindow>

class CameraPanel : public QWidget
{
    Q_OBJECT

public:
    CameraPanel(QMainWindow *parent);

    QMainWindow *mainWindow;
};

#endif // CAMERAPANEL_H
