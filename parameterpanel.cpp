#include "parameterpanel.h"
#include "mainwindow.h"

ParameterDialog::ParameterDialog(QMainWindow *parent) : PanelDialog(parent)
{
    mainWindow = parent;
    setWindowTitle("Set Parameter");
    panel = new ParameterPanel(mainWindow);
    QVBoxLayout *layout = new QVBoxLayout();
    layout->addWidget(panel);
    setLayout(layout);
}

void ParameterDialog::show()
{
    panel->initialize();
    QDialog::show();
}

ParameterPanel::ParameterPanel(QMainWindow *parent) : QWidget(parent)
{
    mainWindow = parent;

    options = new QComboBox(mainWindow);
    for (int i = 0; i < NUM_OPTIONS; i++) {
        if (MW->co->options[i].flags != OPT_EXIT)
            options->addItem(MW->co->options[i].help, QVariant(i));
    }
    parameter = new QLineEdit();
    QLabel *lbl00 = new QLabel("Command Line Equivalent: ");
    cmd_line_equiv = new QLabel();
    cmd_line_equiv->setStyleSheet("QLabel { background-color : lightGray; font : 10pt 'Courier'; }");
    cmd_line_equiv->setText("This is a test");
    QPushButton *set = new QPushButton("Set");
    QPushButton *clear = new QPushButton("Clear");
    QGridLayout *layout = new QGridLayout();
    layout->addWidget(options,         0, 0, 1, 2);
    layout->addWidget(parameter,       1, 0, 1, 2);
    layout->addWidget(lbl00,           2, 0, 1, 2);
    layout->addWidget(cmd_line_equiv,  3, 0, 1, 2);
    layout->addWidget(clear,           4, 0, 1, 1, Qt::AlignCenter);
    layout->addWidget(set,             4, 1, 1, 1, Qt::AlignCenter);

    layout->setRowStretch(3, 10);

    setLayout(layout);

    connect(set, SIGNAL(clicked()), this, SLOT(set()));
    connect(clear, SIGNAL(clicked()), this, SLOT(clear()));
}

void ParameterPanel::initialize()
{
    QString str;

    if (MW->co->start_time > 0) {
        int arg = MW->co->start_time;
        arg /= 1000000;
        QString start_time = QString::number(arg);

        QTextStream(&str) << " -ss " << start_time;
    }

    if (MW->co->duration > 0) {
        int arg = MW->co->duration;
        arg /= 1000000;
        QString duration = QString::number(arg);

        QTextStream(&str) << " -t " << duration;
    }

    if (MW->co->nb_vfilters > 0) {
        QTextStream(&str) << " -vf";
        for (int i = 0; i < MW->co->nb_vfilters; i++) {
            QTextStream(&str) << " " << MW->co->vfilters_list[i];
        }
    }
    cmd_line_equiv->setText(str);
}

void ParameterPanel::set()
{
    int option_index = options->currentData().toInt();
    QString option_name = MW->co->options[option_index].name;
    //cmd_line_equiv->setText("-" + option_name + " " + parameter->text());
    MW->co->options[option_index].u.func_arg(NULL, NULL, parameter->text().toLatin1().data());
    initialize();

    //cout << parameter->text().toStdString() << endl;
    //MW->co->opt_add_vfilter(NULL, NULL, parameter->text().toLatin1().data());
}

void ParameterPanel::clear()
{
    parameter->setText("");
    int option_index = options->currentData().toInt();
    QString option_name = MW->co->options[option_index].name;
    cmd_line_equiv->setText("");
    MW->co->options[option_index].u.func_arg(NULL, NULL, parameter->text().toLatin1().data());
}
