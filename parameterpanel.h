#ifndef PARAMETERPANEL_H
#define PARAMETERPANEL_H

#include <QMainWindow>
#include <QLabel>
#include <QComboBox>
#include <QLineEdit>

#include "Utilities/paneldialog.h"

class OptionBox : public QComboBox
{
public:
    OptionBox(QWidget *parent = nullptr);
    bool keyInput = false;

protected:
    void keyPressEvent(QKeyEvent *event);
    void mousePressEvent(QMouseEvent *event);
};

class ParameterPanel : public QWidget
{
    Q_OBJECT

public:
    ParameterPanel(QMainWindow *parent);
    void setCmdLine();

    QMainWindow *mainWindow;
    OptionBox *options;
    QLineEdit *parameter;
    QLabel *cmd_line_equiv;

public slots:
    void set();
    void clear();
    void optionChanged(int);
    void parameterEntered();
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
