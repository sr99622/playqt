#ifndef PANEL_H
#define PANEL_H

#include <QMainWindow>

class Panel : public QWidget
{
    Q_OBJECT

public:
    Panel(QMainWindow *parent);

    virtual void saveSettings();
    bool changed = false;

    QMainWindow *mainWindow;

};

#endif // PANEL_H
