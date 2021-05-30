#ifndef MAINPANEL_H
#define MAINPANEL_H

#include <QMainWindow>
#include <QLabel>

#include "controlpanel.h"
#include "Utilities/displaycontainer.h"

class MainPanel : public QWidget
{
    Q_OBJECT

public:
    MainPanel(QMainWindow *parent);
    void resizeEvent(QResizeEvent *event) override;

    QMainWindow *mainWindow;
    ControlPanel *controlPanel;
    DisplayContainer *displayContainer;
    QLabel *label;

};

#endif // MAINPANEL_H
