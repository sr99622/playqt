#ifndef OPTIONPANEL_H
#define OPTIONPANEL_H

#include <QMainWindow>
#include <QTextEdit>
#include <QFont>
#include <QLineEdit>
#include <QLabel>
#include <QRunnable>
#include <QComboBox>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavutil/mem.h"
#include "libavutil/avassert.h"
#include "libavutil/parseutils.h"
#include "libavutil/samplefmt.h"
}

#include "Utilities/paneldialog.h"
#include "Ffplay/CommandOptions.h"
#include "parameterpanel.h"
class OptionPanel : public QWidget
{
    Q_OBJECT

public:
    OptionPanel(QMainWindow *parent);

    QTextEdit *helpDisplay;
    QStringList optionList;
    QLineEdit *filterEdit;
    QLabel *currentOption;

    void display(const QString& str, const QString& title);
    const QString getOptionList();
    const QString filterOptionList(const QString& filter);

    char get_media_type_char(enum AVMediaType type);
    static int compare_codec_desc(const void *a, const void *b);
    const AVCodec *next_codec_for_id(enum AVCodecID id, const AVCodec *prev, int encoder);
    unsigned get_codecs_sorted(const AVCodecDescriptor ***rcodecs);
    const QString print_codecs(int encoder);
    const QString show_filters();
    const QString show_protocols();
    const QString show_bsfs();
    const QString show_colors();
    const QString show_pix_fmts();
    const QString show_layouts();
    const QString show_sample_fmts();
    void show_help_codec(const char *name, int encoder);
    const QString print_codec(const AVCodec *c);
    const QString show_formats_devices(int device_only, int muxdemuxers);
    int is_device(const AVClass *avclass);
    const QString show_formats();
    void show_muxers();
    const QString show_demuxers();
    const QString show_devices();
    const QString show_help_options(const OptionDef *options, const char *msg, int req_flags, int rej_flags, int alt_flags);

    QMainWindow *mainWindow;
    ParameterDialog *parameterDialog;

public slots:
    void test();
    void clear();
    void close();
    void filterChanged(const QString&);
    void decoders();
    void filters();
    void bsfs();
    void pix_fmts();
    void formats();
    void sample_fmts();
    void protocols();
    void layouts();
    void colors();
    void demuxers();
    void devices();
    void help();
    void details();
    void config();
    void showConfig(const QString&);
    void showParameterDialog();

};

class OptionDialog : public PanelDialog
{
    Q_OBJECT

public:
    OptionDialog(QMainWindow *parent);
    int getDefaultWidth() override;
    int getDefaultHeight() override;

    QMainWindow *mainWindow;
    OptionPanel *panel;

    int defaultWidth = 520;
    int defaultHeight = 640;

};



#endif // OPTIONPANEL_H
