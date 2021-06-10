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
    panel->setCmdLine();
    QDialog::show();
}

OptionBox::OptionBox(QWidget *parent) : QComboBox(parent)
{

}

void OptionBox::keyPressEvent(QKeyEvent *event)
{
    keyInput = true;
    QComboBox::keyPressEvent(event);
}

void OptionBox::mousePressEvent(QMouseEvent *event)
{
    keyInput = false;
    QComboBox::mousePressEvent(event);
}

ParameterPanel::ParameterPanel(QMainWindow *parent) : QWidget(parent)
{
    mainWindow = parent;

    options = new OptionBox(mainWindow);
    for (int i = 0; i < NUM_OPTIONS; i++) {
        if (!(MW->co->options[i].flags & OPT_EXIT))
            options->addItem(MW->co->options[i].help, QVariant(i));
    }
    parameter = new QLineEdit();
    QLabel *lbl00 = new QLabel("Command Line Equivalent: ");
    cmd_line_equiv = new QLabel();
    cmd_line_equiv->setStyleSheet("QLabel { background-color : lightGray; font : 10pt 'Courier'; }");
    cmd_line_equiv->setText("This is a test");
    QPushButton *set = new QPushButton("Set");
    QPushButton *clear = new QPushButton("Clear");
    clear->setFocusPolicy(Qt::NoFocus);
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
    connect(options, SIGNAL(currentIndexChanged(int)), this, SLOT(optionChanged(int)));
    connect(parameter, SIGNAL(returnPressed()), this, SLOT(parameterEntered()));
}

void ParameterPanel::setCmdLine()
{
    QString str;

    if (QString(MW->co->video_codec_name).length() > 0) {
        QTextStream(&str) << " -vcodec " << MW->co->video_codec_name;
    }

    if (MW->co->audio_disable > 0) {
        QTextStream(&str) << " -an";
    }

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

void ParameterPanel::parameterEntered()
{
    set();
    options->setFocus();
}

void ParameterPanel::optionChanged(int index)
{
    parameter->setText("");
    if (!options->keyInput)
        parameter->setFocus();
}

void ParameterPanel::set()
{
    int index = options->currentData().toInt();
    QString name = MW->co->options[index].name;

    cout << "name: " << name.toStdString() << endl;

    if (MW->co->options[index].u.dst_ptr == 0)
        cout << "null ptr" << endl;

    if (!strcmp(MW->co->options[index].name, "vcodec")) {
        cout << "vcodec" << endl;
        // believe it or not, the address of video_codec_name changes after program starts running main()
        if (parameter->text().length() == 0)
            MW->co->video_codec_name = 0;
        else
            MW->co->video_codec_name = av_strdup(parameter->text().toLatin1().data());
    }
    else if (MW->co->options[index].flags & OPT_FUNC) {
        cout << "func_arg: " << parameter->text().toLatin1().data() << endl;
        MW->co->options[index].u.func_arg(NULL, NULL, parameter->text().toLatin1().data());
    }
    else {
        if (MW->co->options[index].flags & OPT_BOOL) {
            cout << "opt bool" << endl;
            *(int *)MW->co->options[index].u.dst_ptr = parse_number_or_die(MW->co->options[index].name, "1", OPT_INT64, INT_MIN, INT_MAX);
        }
    }

    cout << "test 1" << endl;
    setCmdLine();
}

void ParameterPanel::clear()
{
    parameter->setText("");
    MW->co->duration = AV_NOPTS_VALUE;
    MW->co->start_time = AV_NOPTS_VALUE;
    MW->co->opt_add_vfilter(NULL, NULL, "");
    MW->co->video_codec_name = 0;
    MW->co->audio_disable = 0;
    setCmdLine();
}
