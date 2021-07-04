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
    int tick;

protected:
    void mouseMoveEvent(QMouseEvent *e) override;
    void mousePressEvent(QMouseEvent *e) override;
    void mouseReleaseEvent(QMouseEvent *e) override;
    void keyPressEvent(QKeyEvent *e) override;
    bool event(QEvent *e) override;

public slots:
    void moved(int value);
    void pressed();
    void released();
};

#endif // DISPLAYSLIDER_H
