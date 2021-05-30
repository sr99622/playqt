#ifndef DISPLAYSLIDER_H
#define DISPLAYSLIDER_H

#include <QMainWindow>
#include <QSlider>

class DisplaySlider : public QSlider
{
    Q_OBJECT

public:
    DisplaySlider(QMainWindow *parent);

    QMainWindow *mainWindow;
    bool previously_paused;

protected:
    void mouseReleaseEvent(QMouseEvent *e) override;
    void keyPressEvent(QKeyEvent *e) override;


public slots:
    void moved(int value);
    void pressed();
    void released();
};

#endif // DISPLAYSLIDER_H
