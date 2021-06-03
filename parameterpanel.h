#ifndef PARAMETERPANEL_H
#define PARAMETERPANEL_H

#include <QMainWindow>
#include <QLabel>
#include <QComboBox>
#include <QLineEdit>

#include "Utilities/paneldialog.h"

class ParameterPanel : public QWidget
{
    Q_OBJECT

public:
    ParameterPanel(QMainWindow *parent);
    void initialize();

    QMainWindow *mainWindow;
    QComboBox *options;
    QLineEdit *parameter;
    QLabel *cmd_line_equiv;

public slots:
    void set();
    void clear();
};

class ParameterDialog : public PanelDialog
{
    Q_OBJECT

public:
    ParameterDialog(QMainWindow *parent);
    void show();

    QMainWindow *mainWindow;
    ParameterPanel *panel;
};

#endif // PARAMETERPANEL_H
