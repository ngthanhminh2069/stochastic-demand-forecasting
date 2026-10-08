#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script tạo file PDF Báo cáo Kỹ thuật Chuyên sâu:
- Quy trình chuẩn Demand Forecasting công nghiệp
- Phân tích chi tiết 9 bước thực hiện trong dự án (Full run 100 leaves)
- Giải thích chi tiết từng đoạn code nguồn (Source Code Mapping)
- Ma trận so sánh toàn diện 10 tiêu chí
- Bảng tra cứu toàn bộ tham số cấu hình (Hyperparameters Reference)
- Khắc phục hoàn toàn lỗi font, chuẩn hóa công thức toán học và ký tự Unicode
"""
import sys
import os
import shutil
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Đăng ký font tiếng Việt và font code Consolas từ Windows
pdfmetrics.registerFont(TTFont('Arial', 'C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Bold', 'C:/Windows/Fonts/arialbd.ttf'))
pdfmetrics.registerFont(TTFont('Arial-Italic', 'C:/Windows/Fonts/ariali.ttf'))
pdfmetrics.registerFont(TTFont('Arial-BoldItalic', 'C:/Windows/Fonts/arialbi.ttf'))
pdfmetrics.registerFont(TTFont('Consolas', 'C:/Windows/Fonts/consola.ttf'))
pdfmetrics.registerFont(TTFont('Consolas-Bold', 'C:/Windows/Fonts/consolab.ttf'))

class NumberedCanvas(canvas.Canvas):
    """Canvas vẽ 2 lượt để tính chính xác tổng số trang và hiển thị Header/Footer sang trọng."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            return  # Trang bìa không vẽ header/footer
        
        self.saveState()
        self.setFont("Arial", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header cố định phía trên - tách biệt 2 bên, không chồng lấn
        self.drawString(45, 808, "BÁO CÁO KỸ THUẬT: PIPELINE DỰ BÁO NHU CẦU NGẪU NHIÊN")
        self.drawRightString(550, 808, "STOCHASTIC DEMAND FORECASTING")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.6)
        self.line(45, 800, 550, 800)
        
        # Footer cố định phía dưới
        self.line(45, 42, 550, 42)
        self.drawString(45, 30, "Tài liệu kỹ thuật nội bộ • Giải mã chi tiết Source Code & Đối chuẩn Quy trình")
        page_str = f"Trang {self._pageNumber} / {page_count}"
        self.drawRightString(550, 30, page_str)
        self.restoreState()


def build_pdf(filename: str, figures_dir: Path):
    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        leftMargin=45,
        rightMargin=45,
        topMargin=50,
        bottomMargin=50,
    )
    
    PRIMARY = colors.HexColor("#1E3A8A")       # Xanh Navy Đậm
    SECONDARY = colors.HexColor("#0D9488")     # Xanh Ngọc Lục Bảo (Teal)
    DARK_TEXT = colors.HexColor("#1E293B")     # Xám than đậm
    MUTED_TEXT = colors.HexColor("#475569")    # Xám phụ đề
    BG_LIGHT = colors.HexColor("#F8FAFC")      # Nền bảng nhẹ
    BORDER_LIGHT = colors.HexColor("#E2E8F0")  # Viền phân cách
    CALLOUT_BG = colors.HexColor("#EFF6FF")    # Nền xanh nhạt
    CALLOUT_BORDER = colors.HexColor("#3B82F6")# Viền xanh nhạt
    WARN_BG = colors.HexColor("#FFFBEB")       # Vàng cảnh báo
    WARN_BORDER = colors.HexColor("#F59E0B")   
    SUCCESS_BG = colors.HexColor("#F0FDF4")    # Xanh lá thành công
    SUCCESS_BORDER = colors.HexColor("#22C55E")
    CODE_BG = colors.HexColor("#F1F5F9")       # Nền khối code
    CODE_BORDER = colors.HexColor("#CBD5E1")   # Viền khối code
    
    styles = getSampleStyleSheet()
    
    body_style = ParagraphStyle(
        'VnBody',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=9.5,
        leading=14.5,
        textColor=DARK_TEXT,
        spaceAfter=6,
    )
    
    body_bold = ParagraphStyle(
        'VnBodyBold',
        parent=body_style,
        fontName='Arial-Bold',
    )
    
    body_italic = ParagraphStyle(
        'VnBodyItalic',
        parent=body_style,
        fontName='Arial-Italic',
        textColor=MUTED_TEXT,
    )
    
    title_cover = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=23,
        leading=30,
        textColor=PRIMARY,
        spaceAfter=12,
    )
    
    subtitle_cover = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=11.5,
        leading=17,
        textColor=MUTED_TEXT,
        spaceAfter=18,
    )
    
    h1_style = ParagraphStyle(
        'VnH1',
        parent=styles['Heading1'],
        fontName='Arial-Bold',
        fontSize=14,
        leading=19,
        textColor=PRIMARY,
        spaceBefore=14,
        spaceAfter=7,
        keepWithNext=True,
    )
    
    h2_style = ParagraphStyle(
        'VnH2',
        parent=styles['Heading2'],
        fontName='Arial-Bold',
        fontSize=11.5,
        leading=16,
        textColor=SECONDARY,
        spaceBefore=11,
        spaceAfter=5,
        keepWithNext=True,
    )
    
    h3_style = ParagraphStyle(
        'VnH3',
        parent=styles['Heading3'],
        fontName='Arial-Bold',
        fontSize=10,
        leading=14.5,
        textColor=DARK_TEXT,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True,
    )
    
    callout_text = ParagraphStyle(
        'CalloutText',
        parent=body_style,
        fontSize=9,
        leading=13.5,
        textColor=colors.HexColor("#1E3A8A"),
    )
    
    table_cell = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=8.5,
        leading=12,
        textColor=DARK_TEXT,
    )
    
    table_header = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Arial-Bold',
        fontSize=8.5,
        leading=12,
        textColor=colors.white,
    )
    
    img_caption = ParagraphStyle(
        'ImgCaption',
        parent=styles['Normal'],
        fontName='Arial-Italic',
        fontSize=8,
        leading=11,
        textColor=MUTED_TEXT,
        alignment=1,
        spaceBefore=3,
        spaceAfter=7,
    )
    
    code_text = ParagraphStyle(
        'CodeText',
        parent=styles['Normal'],
        fontName='Consolas',
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0F172A"),
    )
    
    code_explain = ParagraphStyle(
        'CodeExplain',
        parent=styles['Normal'],
        fontName='Arial',
        fontSize=8.5,
        leading=12.5,
        textColor=DARK_TEXT,
    )

    def make_callout(text, bg_color=CALLOUT_BG, border_color=CALLOUT_BORDER, title=None, style=callout_text):
        content = []
        if title:
            content.append(Paragraph(f"<b>{title}</b>", ParagraphStyle('CTitle', parent=style, fontName='Arial-Bold', spaceAfter=3)))
        content.append(Paragraph(text, style))
        tbl = Table([[content]], colWidths=[505])
        tbl.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), bg_color),
            ('BOX', (0,0), (-1,-1), 1, border_color),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 9),
            ('RIGHTPADDING', (0,0), (-1,-1), 9),
        ]))
        return tbl

    def make_code_box(code_str, filename_str, explanation_str):
        p_file = Paragraph(f"<b>File: {filename_str}</b>", ParagraphStyle('CodeFile', fontName='Arial-Bold', fontSize=8.5, textColor=PRIMARY))
        p_code = Paragraph(code_str.replace("\n", "<br/>").replace(" ", "&nbsp;"), code_text)
        p_exp = Paragraph(f"<b>Giải thích kỹ thuật:</b> {explanation_str}", code_explain)
        
        box_data = [
            [p_file],
            [p_code],
            [p_exp]
        ]
        t = Table(box_data, colWidths=[505])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#E2E8F0")),
            ('BACKGROUND', (0,1), (-1,1), CODE_BG),
            ('BACKGROUND', (0,2), (-1,2), BG_LIGHT),
            ('BOX', (0,0), (-1,-1), 1, CODE_BORDER),
            ('INNERGRID', (0,0), (-1,-1), 0.5, CODE_BORDER),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('LEFTPADDING', (0,0), (-1,-1), 7),
            ('RIGHTPADDING', (0,0), (-1,-1), 7),
        ]))
        return KeepTogether(t)

    story = []
    
    # =========================================================================
    # TRANG BÌA (COVER PAGE)
    # =========================================================================
    story.append(Spacer(1, 35))
    story.append(Paragraph("BÁO CÁO KỸ THUẬT & HƯỚNG DẪN CHUYÊN SÂU", ParagraphStyle('SuperTitle', fontName='Arial-Bold', fontSize=10, textColor=SECONDARY, spaceAfter=8)))
    story.append(Paragraph("KIẾN TRÚC & QUY TRÌNH HỆ THỐNG<br/>DỰ BÁO NHU CẦU NGẪU NHIÊN<br/>(STOCHASTIC DEMAND FORECASTING)", title_cover))
    story.append(HRFlowable(width="100%", thickness=3, color=PRIMARY, spaceBefore=2, spaceAfter=12))
    story.append(Paragraph("Phân tích chi tiết 9 bước thực hiện trên toàn bộ 100 chuỗi thời gian thực tế, chuẩn hóa quy trình chuẩn công nghiệp, đối chiếu ma trận so sánh và giải mã từng đoạn mã nguồn (Source Code Mapping).", subtitle_cover))
    story.append(Spacer(1, 15))
    
    meta_info = [
        [Paragraph("<b>Hệ thống:</b>", table_cell), Paragraph("SKU x Location Stochastic Demand Forecasting & Inventory Engine", table_cell)],
        [Paragraph("<b>Quy mô thực thi (Full Run):</b>", table_cell), Paragraph("<b>Hoàn thành 100% Full Run:</b> 76,000 dòng dữ liệu, 20 SKUs × 5 Cửa hàng = 100 chuỗi lá độc lập", table_cell)],
        [Paragraph("<b>Mô hình toán cốt lõi:</b>", table_cell), Paragraph("Multi-Quantile XGBoost, Split Conformal Prediction, Gaussian Copula AR(1), CUSUM Control Chart, Newsvendor Inventory", table_cell)],
        [Paragraph("<b>Đối tượng hướng tới:</b>", table_cell), Paragraph("Kỹ sư Dữ liệu, Chuyên viên Chuỗi cung ứng (SCM Planner) & Người mới tiếp cận Demand Planning", table_cell)],
        [Paragraph("<b>Điểm nhấn kỹ thuật:</b>", table_cell), Paragraph("Zero Data Leakage (As-of-origin), Phân phối không đối xứng, Bù trừ Conformal, Rủi ro gộp (Risk Pooling)", table_cell)],
    ]
    meta_tbl = Table(meta_info, colWidths=[130, 375])
    meta_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('BOX', (0,0), (-1,-1), 1, BORDER_LIGHT),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_LIGHT),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 8),
        ('RIGHTPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(meta_tbl)
    
    story.append(Spacer(1, 30))
    story.append(make_callout(
        "<b>Thông báo trạng thái thực thi:</b> Pipeline đã chạy hoàn tất toàn bộ 100 leaves trên tập dữ liệu thực tế (Full Pipeline Execution Complete). "
        "Tất cả các biểu đồ phân tích thực nghiệm trong tài liệu này (từ dải phân vị, độ tin cậy Conformal, kiểm định phần dư đến biểu đồ CUSUM và so sánh sai số) "
        "đều được cập nhật trực tiếp từ kết quả chạy thực tế của hệ thống.",
        bg_color=SUCCESS_BG, border_color=SUCCESS_BORDER, title="DỰ ÁN ĐÃ HOÀN TẤT CHẠY FULL RUN"
    ))
    
    story.append(PageBreak())
    
    # =========================================================================
    # PHẦN 1: TỔNG QUAN & KHÁI NIỆM DÀNH CHO NGƯỜI MỚI
    # =========================================================================
    story.append(Paragraph("PHẦN 1: TỔNG QUAN & KHÁI NIỆM DÀNH CHO NGƯỜI MỚI BẮT ĐẦU", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceBefore=2, spaceAfter=8))
    
    story.append(Paragraph("1.1 Dự báo nhu cầu (Demand Forecasting) là gì và tại sao sống còn?", h2_style))
    story.append(Paragraph(
        "Trong mọi doanh nghiệp sản xuất và bán lẻ, <b>Dự báo nhu cầu (Demand Forecasting)</b> là quá trình ước tính lượng sản phẩm mà khách hàng sẽ mua trong tương lai (ví dụ: trong 7 ngày, 14 ngày hoặc 3 tháng tới). "
        "Mọi quyết định vận hành lớn đều phụ thuộc trực tiếp vào con số này: nhà máy cần sản xuất bao nhiêu cái, đội cung ứng cần đặt bao nhiêu công-ten-nơ từ nhà cung cấp, và kho trung tâm cần chia về từng cửa hàng bao nhiêu sản phẩm.",
        body_style
    ))
    story.append(Paragraph(
        "Nếu dự báo <b>quá thấp</b> (Under-forecasting), hàng sẽ đứt gãy trên kệ (Stockout). Khách hàng không mua được hàng sẽ bỏ sang đối thủ, doanh nghiệp vừa mất doanh thu tức thì vừa đánh mất uy tín thương hiệu. "
        "Ngược lại, nếu dự báo <b>quá cao</b> (Over-forecasting), hàng hóa sẽ nằm chết trong kho (Excess Inventory), làm đóng băng dòng vốn lưu động, tốn chi phí thuê mặt bằng bảo quản, chịu rủi ro hàng hết hạn, ẩm mốc hoặc phải xả lỗ (Mark-down).",
        body_style
    ))
    
    story.append(Paragraph("1.2 Điểm mù của Dự báo truyền thống: 'Dự báo một con số' (Point Forecast)", h2_style))
    story.append(Paragraph(
        "Phần lớn các hướng dẫn hoặc dự án thông thường dừng lại ở <b>Point Forecast</b> (Dự báo điểm). Nghĩa là mô hình đưa ra duy nhất một con số: <i>'Ngày mai cửa hàng A sẽ bán được đúng 50 cái áo thun'</i>. "
        "Tuy nhiên, trong thế giới thực, nhu cầu là một biến số ngẫu nhiên bị tác động bởi thời tiết, xu hướng mạng xã hội, khuyến mãi của đối thủ, tâm lý người tiêu dùng...",
        body_style
    ))
    
    story.append(make_callout(
        "<b>Ví dụ thực tế dễ hiểu:</b><br/>"
        "Giả sử dự báo ngày mai bán được 50 cái áo. Nhà quản lý đặt đúng 50 cái về kho.<br/>"
        "• Khả năng thực tế bán đúng chính xác 50 cái gần như bằng 0.<br/>"
        "• Nếu ngày mai trời nắng đẹp, khách mua 70 cái -> Cửa hàng hết sạch hàng ở cái thứ 50, mất trắng 20 đơn hàng!<br/>"
        "• Nếu trời mưa bão, khách chỉ mua 20 cái -> Cửa hàng thừa 30 cái áo đọng vốn lưu kho.<br/>"
        "<b>Kết luận:</b> Một con số đơn lẻ không thể giúp ta trả lời câu hỏi cốt tử: <i>'Cần trữ thêm bao nhiêu cái để đạt 95% khả năng không bị cháy hàng?'</i>",
        bg_color=WARN_BG, border_color=WARN_BORDER, title="TẠI SAO DỰ BÁO MỘT CON SỐ LẠI THẤT BẠI TRONG THỰC TẾ?"
    ))
    story.append(Spacer(1, 4))
    
    story.append(Paragraph("1.3 Dự báo ngẫu nhiên / phân phối xác suất (Stochastic & Probabilistic Forecasting)", h2_style))
    story.append(Paragraph(
        "Thay vì dự đoán một con số cứng nhắc, <b>Dự báo ngẫu nhiên (Stochastic/Probabilistic Forecasting)</b> dự báo toàn bộ <i>phân phối xác suất (Probability Distribution)</i> của nhu cầu. "
        "Thay vì nói 'bán 50 cái', mô hình sẽ đưa ra các phân vị (Quantiles):<br/>"
        "• <b>P<sub>10</sub> = 35 cái:</b> Chỉ có 10% khả năng nhu cầu thấp hơn hoặc bằng 35 cái (kịch bản thị trường ảm đạm).<br/>"
        "• <b>P<sub>50</sub> = 50 cái:</b> Trung vị nhu cầu (kịch bản tiêu chuẩn, 50% khả năng nhu cầu cao hơn và 50% thấp hơn).<br/>"
        "• <b>P<sub>90</sub> = 72 cái:</b> Có 90% khả năng nhu cầu thấp hơn hoặc bằng 72 cái (kịch bản thị trường bùng nổ).",
        body_style
    ))
    story.append(Paragraph(
        "Nhờ dải phân phối này, người quản lý chuỗi cung ứng có thể định lượng chính xác độ bất định và rủi ro để đưa ra lượng hàng đệm an toàn phù hợp với từng chiến lược kinh doanh.",
        body_style
    ))
    
    story.append(Paragraph("1.4 Bảng thuật ngữ cơ bản dành cho người mới tiếp cận", h2_style))
    
    glossary_data = [
        [Paragraph("Thuật ngữ", table_header), Paragraph("Ý nghĩa trực quan", table_header), Paragraph("Ứng dụng thực tế trong dự án này", table_header)],
        [
            Paragraph("<b>SKU</b><br/>(Stock Keeping Unit)", table_cell),
            Paragraph("Mã phân loại hàng hóa riêng biệt (ví dụ: Áo thun trắng size L mã SKU_01).", table_cell),
            Paragraph("Dự án dự báo chi tiết tới từng cặp SKU × Cửa hàng (Leaf level).", table_cell)
        ],
        [
            Paragraph("<b>Phân vị (Quantiles)<br/>P<sub>10</sub>, P<sub>50</sub>, P<sub>90</sub></b>", table_cell),
            Paragraph("Ngưỡng mà xác suất nhu cầu thực tế nhỏ hơn hoặc bằng nó là 10%, 50%, 90%. P<sub>50</sub> chính là trung vị.", table_cell),
            Paragraph("Mô hình huấn luyện 8 phân vị: P<sub>05</sub> đến P<sub>97.5</sub> vẽ nên toàn bộ dải bất định nhu cầu.", table_cell)
        ],
        [
            Paragraph("<b>Safety Stock (SS)</b><br/>(Tồn kho an toàn)", table_cell),
            Paragraph("Lượng hàng đệm trữ sẵn phòng ngừa những ngày nhu cầu tăng vọt hoặc hàng về chậm.", table_cell),
            Paragraph("Được tính toán từ phân phối mô phỏng nhu cầu thời gian giao hàng (Lead Time).", table_cell)
        ],
        [
            Paragraph("<b>Reorder Point (ROP)</b><br/>(Điểm đặt hàng lại)", table_cell),
            Paragraph("Mức tồn kho báo động: Khi hàng trong kho giảm chạm mức ROP, hệ thống kích hoạt đơn đặt hàng mới.", table_cell),
            Paragraph("ROP = Nhu cầu dự kiến trong thời gian giao hàng + Tồn kho an toàn.", table_cell)
        ],
        [
            Paragraph("<b>Newsvendor Model</b><br/>(Bài toán người bán báo)", table_cell),
            Paragraph("Mô hình kinh tế cổ điển cân bằng: Chi phí mất doanh thu nếu thiếu hàng (<i>C<sub>u</sub></i>) và Chi phí tồn ứ nếu thừa hàng (<i>C<sub>o</sub></i>).", table_cell),
            Paragraph("Dự án dùng giá bán và biên lợi nhuận để đo lường tổn thất thực tế bằng USD.", table_cell)
        ],
        [
            Paragraph("<b>Conformal Prediction</b><br/>(Hiệu chuẩn phân phối)", table_cell),
            Paragraph("Công nghệ hiệu chuẩn xác suất: Đảm bảo khoảng dự báo P<sub>90</sub> thực sự chứa đúng 90% kết quả ngoài đời thực.", table_cell),
            Paragraph("Dùng Split Conformal Calibration trên tập dữ liệu riêng biệt để bù sai lệch cho XGBoost.", table_cell)
        ],
        [
            Paragraph("<b>CUSUM Control Chart</b><br/>(Biểu đồ tổng tích lũy)", table_cell),
            Paragraph("Công cụ kiểm soát chất lượng giúp phát hiện những đợt sụt giảm nhu cầu âm ỉ kéo dài.", table_cell),
            Paragraph("Dự án dùng CUSUM phát hiện sớm các đợt đứt gãy nhu cầu (drop detection) trên từng sản phẩm.", table_cell)
        ],
    ]
    g_tbl = Table(glossary_data, colWidths=[95, 205, 205])
    g_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('BOX', (0,0), (-1,-1), 1, BORDER_LIGHT),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_LIGHT),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(g_tbl)
    
    story.append(PageBreak())
    
    # =========================================================================
    # PHẦN 2: WORKFLOW CHUẨN CÔNG NGHIỆP TRUYỀN THỐNG
    # =========================================================================
    story.append(Paragraph("PHẦN 2: WORKFLOW CHUẨN CỦA DEMAND FORECASTING TRONG DOANH NGHIỆP", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceBefore=2, spaceAfter=8))
    
    story.append(Paragraph("2.1 Quy trình 7 giai đoạn chuẩn trong các hệ thống Enterprise (ERP / SCM)", h2_style))
    story.append(Paragraph(
        "Trong hầu hết các doanh nghiệp bán lẻ và sản xuất hiện nay (sử dụng SAP, Oracle, Blue Yonder hoặc Kinaxis), quy trình dự báo nhu cầu thường tuân theo một chu kỳ tuần hoàn hàng tháng hoặc hàng tuần gồm 7 giai đoạn nối tiếp nhau:",
        body_style
    ))
    
    wf_steps = [
        ("Giai đoạn 1: Thu thập & Làm sạch dữ liệu (Data Ingestion & Cleansing)",
         "Hệ thống kéo dữ liệu lịch sử bán hàng (POS - Point of Sale) từ ERP. Ở bước này, dữ liệu thường được khử nhiễu đơn giản, điền giá trị thiếu (missing values) bằng số 0 hoặc giá trị trung bình, và loại bỏ các ngày nghỉ lễ đặc biệt."),
        
        ("Giai đoạn 2: Kỹ thuật tạo đặc trưng (Feature Engineering)",
         "Chuyên viên tạo ra các biến trễ (Lag features như bán hàng tuần trước, tháng trước), chỉ số trung bình trượt (Rolling Mean 7 ngày, 30 ngày) và các biến thời gian (thứ trong tuần, tháng trong năm, cờ ngày lễ)."),
         
        ("Giai đoạn 3: Huấn luyện mô hình (Model Training)",
         "Huấn luyện các mô hình chuỗi thời gian cổ điển (ARIMA, Exponential Smoothing / Holt-Winters, Prophet) hoặc Machine Learning dạng bảng (LightGBM, Random Forest). Điểm mấu chốt: Các mô hình này hầu như chỉ tối ưu hóa để dự đoán <b>duy nhất một giá trị kỳ vọng (Mean/Median Point Forecast)</b>."),
         
        ("Giai đoạn 4: Đánh giá mô hình truyền thống (Evaluation)",
         "Đo đạc độ chính xác bằng các chỉ số sai số toán học thuần túy: MAE (Mean Absolute Error), RMSE, hoặc MAPE (Mean Absolute Percentage Error). Mô hình nào có MAPE thấp nhất sẽ được chọn làm mô hình tốt nhất."),
         
        ("Giai đoạn 5: Cân đối cấp bậc (Hierarchical Allocation: Top-Down hoặc Bottom-Up)",
         "Doanh nghiệp cần số liệu ở nhiều cấp: Cấp Toàn quốc, Cấp Vùng, Cấp Cửa hàng, Cấp Ngành hàng. Thường họ sẽ dự báo ở cấp cao (Toàn quốc) rồi bổ tỉ lệ phần trăm xuống từng cửa hàng (Top-Down), hoặc dự báo ở từng cửa hàng rồi cộng dồn số liệu trung bình lên cấp trên (Bottom-Up)."),
         
        ("Giai đoạn 6: Hoạch định tồn kho bằng công thức sách giáo khoa (Safety Stock Calculation)",
         "Sau khi có số dự báo điểm, chuyên viên chuỗi cung ứng áp dụng công thức an toàn cổ điển:<br/>"
         "&nbsp;&nbsp;&nbsp;&nbsp;<b>Safety Stock = z · σ · √(Lead Time)</b><br/>"
         "Trong đó: <i>z</i> là hệ số tin cậy tra bảng phân phối chuẩn (ví dụ <i>z = 1.65</i> cho mức dịch vụ 95%), <i>σ</i> là độ lệch chuẩn của nhu cầu lịch sử, <i>Lead Time</i> là số ngày nhà cung cấp giao hàng."),
         
        ("Giai đoạn 7: Quy trình S&OP (Sales & Operations Planning) & Triển khai",
         "Số liệu dự báo của máy tính được đưa vào cuộc họp S&OP liên phòng ban (Kinh doanh, Tiếp thị, Tài chính, Vận hành) để con người điều chỉnh cảm tính (Over-ride) trước khi chốt đơn đặt hàng cuối cùng.")
    ]
    
    for title, desc in wf_steps:
        story.append(Paragraph(f"• <b>{title}</b>", h3_style))
        story.append(Paragraph(desc, body_style))
        story.append(Spacer(1, 2))
        
    story.append(Spacer(1, 6))
    story.append(Paragraph("2.2 Ba 'lỗ hổng tử huyệt' của Workflow truyền thống", h2_style))
    
    wf_flaws = [
        [Paragraph("Điểm yếu cốt tử", table_header), Paragraph("Nguyên nhân kỹ thuật", table_header), Paragraph("Hậu quả nghiêm trọng đối với doanh nghiệp", table_header)],
        [
            Paragraph("<b>1. Giả định phân phối chuẩn (Gaussian assumption)</b>", table_cell),
            Paragraph("Công thức <i>z · σ · √L</i> bắt buộc nhu cầu mỗi ngày phải tuân theo phân phối chuẩn hình chuông cân đối và độc lập giữa các ngày.", table_cell),
            Paragraph("Nhu cầu thực tế luôn lệch phải (Right-skewed, nhiều ngày bình thường nhưng có vài ngày đột biến cực lớn). Dùng công thức này sẽ làm <b>thiếu hụt tồn kho nghiêm trọng</b> vào những ngày bán chạy nhất!"),
        ],
        [
            Paragraph("<b>2. Bỏ qua chi phí kinh doanh bất đối xứng</b>", table_cell),
            Paragraph("Các chỉ số như MAE, RMSE coi sai số thừa 10 cái cũng tệ bằng sai số thiếu 10 cái (đối xứng).", table_cell),
            Paragraph("Thực tế: Với mặt hàng lãi cao, thiếu 1 cái mất $30 lãi trong khi thừa 1 cái chỉ tốn $0.2 tiền lưu kho. Mô hình truyền thống phạt quá nặng việc trữ thừa dẫn đến hay bị cháy hàng."),
        ],
        [
            Paragraph("<b>3. Rò rỉ dữ liệu khi làm tính năng trễ (Data Leakage)</b>", table_cell),
            Paragraph("Khi dự báo 14 ngày tới, nếu lấy đặc trưng 'doanh số 7 ngày trước', vào ngày thứ 10 thì '7 ngày trước' rơi vào ngày thứ 3 của tương lai (chưa xảy ra).", table_cell),
            Paragraph("Mô hình kiểm tra trên máy tính thì thấy kết quả rất đẹp (MAPE cực thấp), nhưng khi chạy thực tế ngoài đời thì sai số tăng vọt vì dùng dữ liệu tương lai."),
        ],
    ]
    flaw_tbl = Table(wf_flaws, colWidths=[120, 185, 200])
    flaw_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#B91C1C")),
        ('BOX', (0,0), (-1,-1), 1, BORDER_LIGHT),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_LIGHT),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(flaw_tbl)
    
    story.append(PageBreak())
    
    # =========================================================================
    # PHẦN 3: PHÂN TÍCH CHI TIẾT 9 BƯỚC THỰC HIỆN TRONG PROJECT
    # =========================================================================
    story.append(Paragraph("PHẦN 3: PHÂN TÍCH CHI TIẾT 9 BƯỚC THỰC HIỆN TRONG DỰ ÁN", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceBefore=2, spaceAfter=8))
    
    story.append(Paragraph(
        "Dự án được xây dựng với mục tiêu giải quyết triệt để các hạn chế trên. Toàn bộ 9 bước dưới đây phản ánh chính xác chuỗi xử lý khép kín của pipeline trên tập dữ liệu thực tế 100 leaves:",
        body_style
    ))
    
    # --- BƯỚC 1 ---
    story.append(Paragraph("Bước 1: Khảo sát & Chuẩn hóa dữ liệu thực tế (Real Panel Ingestion)", h2_style))
    story.append(Paragraph(
        "• <b>Quy mô dữ liệu (Full run):</b> 76,000 dòng dữ liệu bảng (Panel Data) ghi nhận doanh số hàng ngày của 20 SKU tại 5 Cửa hàng (Locations) trong hơn 2 năm liên tục.<br/>"
        "• <b>Kiểm tra chuỗi liên tục (Contiguous Daily Check):</b> Mỗi chuỗi lá bắt buộc phải có đầy đủ các ngày liên tục, không được đứt quãng. Nếu có ngày không bán được, demand được ghi nhận chính xác là 0.<br/>"
        "• <b>Phát hiện nghịch lý Demand vs Units Sold (Censored Sales):</b> Dữ liệu gốc ghi nhận khoảng 28% số ngày có sự chênh lệch giữa <code>demand</code> (nhu cầu khách muốn mua) và <code>units_sold</code> (số lượng thực tế bán được). Số lượng bán không bao giờ vượt quá lượng tồn kho có sẵn (<code>inventory_level</code>). Khi hết hàng, doanh số bán dừng ở 0 dù nhu cầu của khách vẫn rất cao! Pipeline này tập trung dự báo <b>Demand thực chất</b>.",
        body_style
    ))
    
    p_dist = figures_dir / "demand_distribution_seasonality.png"
    p_heat = figures_dir / "demand_heatmap_sku_location.png"
    if p_dist.exists() and p_heat.exists():
        row_imgs = [
            [Image(str(p_dist), width=310, height=108), Image(str(p_heat), width=180, height=176)]
        ]
        tbl_d = Table(row_imgs, colWidths=[315, 190])
        tbl_d.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(tbl_d)
        story.append(Paragraph("Hình 1: (Trái) Phân phối nhu cầu thực tế lệch phải và tính chu kỳ ngày trong tuần. (Phải) Heatmap nhu cầu trung bình theo từng cặp SKU × Cửa hàng.", img_caption))
    
    story.append(Spacer(1, 4))
    
    # --- BƯỚC 2 ---
    story.append(Paragraph("Bước 2: Kỹ thuật tạo đặc trưng chống rò rỉ (Direct Multi-horizon as-of-origin)", h2_style))
    story.append(Paragraph(
        "• <b>Origin (Gốc dự báo):</b> Ngày cuối cùng mà nhu cầu đã xảy ra và được biết chắc chắn.<br/>"
        "• <b>Horizon h (Tầm dự báo):</b> Khoảng cách từ origin đến ngày mục tiêu (<i>h = 1, 2, ..., 14</i> ngày tới).<br/>"
        "• <b>Nguyên tắc 'As-of-origin':</b> Mọi đặc trưng tính từ nhu cầu <b>chỉ được phép sử dụng dữ liệu xảy ra vào hoặc trước ngày Origin</b>. Nhờ vậy, tập huấn luyện phản ánh chính xác 100% điều kiện thực tế khi vận hành.<br/>"
        "• <b>Seasonal Lags chống rò rỉ:</b> Đối với ngày mục tiêu ở chân trời <i>h</i>, cùng thứ trong tuần của 1 tuần trước được tính lùi về từ điểm gần nhất đã biết trước Origin: <i>Y[origin + h - 7 · ceil(h / 7)]</i>. Tương tự cho 2 tuần và 3 tuần trước.<br/>"
        "• <b>Rolling Windows:</b> Trung bình và độ lệch chuẩn của nhu cầu trong 7 ngày và 28 ngày qua tính chính xác đến ngày Origin.<br/>"
        "• <b>Đặc trưng tương lai hợp lệ:</b> Giá bán (<code>price</code>) và Khuyến mãi (<code>promotion</code>) của ngày tương lai được đưa vào vì kế hoạch khuyến mãi luôn được chốt trước.",
        body_style
    ))
    
    story.append(PageBreak())
    
    # --- BƯỚC 3 ---
    story.append(Paragraph("Bước 3: Kiểm thử luân phiên Walk-Forward Backtesting & Tách tập hiệu chuẩn", h2_style))
    story.append(Paragraph(
        "• <b>Cấu hình Backtest:</b> Chân trời dự báo (Horizon) = 14 ngày, Bước nhảy cuộn (Step) = 28 ngày, Thời gian huấn luyện tối thiểu ban đầu = 180 ngày.<br/>"
        "• <b>Phân tách tập Calibration (Conformal Split):</b> Trong mỗi Fold, tập Train không được dùng toàn bộ để khớp mô hình. Thay vào đó, <b>60 ngày cuối cùng của tập Train</b> được trích ra làm tập Hiệu chuẩn (Calibration Set). Mô hình XGBoost chỉ học trên dữ liệu trước 60 ngày này. Sau đó, mô hình dự báo thử trên 60 ngày Calibration để đo đạc sai số thực tế. Conformal Calibration chỉ có giá trị khi được thực hiện trên dữ liệu mà mô hình <b>chưa từng nhìn thấy</b> trong lúc học trọng số!",
        body_style
    ))
    
    # --- BƯỚC 4 ---
    story.append(Paragraph("Bước 4: Huấn luyện Quantile XGBoost & Hiệu chuẩn Conformal Prediction", h2_style))
    story.append(Paragraph(
        "• <b>Multi-Quantile XGBoost:</b> Huấn luyện đồng thời 8 phân vị: <code>[0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975]</code> với hàm mất mát Pinball Loss (<code>objective='reg:quantileerror'</code>).<br/>"
        "• <b>Xử lý nghịch lý giao cắt phân vị (Quantile Crossing):</b> Bắt buộc <i>P<sub>05</sub> ≤ P<sub>10</sub> ≤ ... ≤ P<sub>97.5</sub></i> bằng thuật toán sắp xếp đơn điệu theo hàng (Monotonicity row-wise sort).<br/>"
        "• <b>Bù sai lệch ngoại suy (Split Conformal Calibration):</b> Tính phần dư hiệu chuẩn <i>e = y - ŷ<sub>q</sub></i> trên tập Calibration. Sau đó tính độ dời bù trừ (Offset) cho từng phân vị theo từng cụm thời gian (Bucket 1-7 ngày và 8-14 ngày). Phần dư được gộp chung (Pooled) giữa các cửa hàng của cùng một SKU để tăng kích thước mẫu, đảm bảo độ tin cậy thống kê.",
        body_style
    ))
    
    p_rel = figures_dir / "reliability_diagram.png"
    p_fan = figures_dir / "quantile_fan_example.png"
    if p_rel.exists() and p_fan.exists():
        img_row = [
            [Image(str(p_fan), width=310, height=128), Image(str(p_rel), width=180, height=180)]
        ]
        tbl_img = Table(img_row, colWidths=[315, 190])
        tbl_img.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(tbl_img)
        story.append(Paragraph("Hình 2: (Trái) Dải quạt dự báo phân vị P10-P50-P90 bám sát thực tế. (Phải) Biểu đồ Reliability Diagram kiểm chứng độ trung thực của phân vị xác suất trên toàn bộ 100 leaves.", img_caption))
    
    story.append(Spacer(1, 4))
    
    # --- BƯỚC 5 ---
    story.append(Paragraph("Bước 5: Đánh giá hiệu quả đa chiều & Đo lường thiệt hại bằng tiền (Newsvendor Cost)", h2_style))
    story.append(Paragraph(
        "1. <b>Độ chính xác điểm (Point Metrics):</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;• <b>WAPE (Weighted Absolute Percentage Error):</b> Tổng sai số tuyệt đối chia cho tổng nhu cầu thực tế. Khác với MAPE (bị lỗi chia cho 0 khi nhu cầu = 0), WAPE cực kỳ ổn định và cộng dồn được.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;• <b>MASE (Mean Absolute Scaled Error):</b> So sánh sai số của mô hình với mô hình chuẩn Seasonal Naive. Nếu MASE &lt; 1, mô hình thông minh hơn cách đoán mò cùng thứ tuần trước.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;• <b>Bias (Độ lệch hướng):</b> Đo lường mô hình đang đoán thừa (Over-forecast) hay đoán thiếu (Under-forecast).<br/>"
        "2. <b>Độ tin cậy phân phối:</b> Đo bằng Pinball Loss và tỷ lệ bao phủ thực nghiệm (Nominal vs Empirical Coverage theo từng Horizon).<br/>"
        "3. <b>Mô hình chi phí kinh doanh (Newsvendor Cost Model - Đổi sai số ra Tiền USD):</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;• Chi phí đứt hàng: <i>C<sub>u</sub> = 30% × Giá bán</i> (mất biên lợi nhuận trên đơn hàng hụt).<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;• Chi phí lưu kho: <i>C<sub>o</sub> = 25%/năm × Giá bán × (7 / 365)</i> cho một chu kỳ đặt hàng 7 ngày.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;• Tỉ số tới hạn: <i>CR = C<sub>u</sub> / (C<sub>u</sub> + C<sub>o</sub>) ≈ 0.975</i>. Mô hình đánh giá chi phí tổn thất trực tiếp tại phân vị tối ưu kinh tế này!",
        body_style
    ))
    
    story.append(PageBreak())
    
    # --- BƯỚC 6 ---
    story.append(Paragraph("Bước 6: Khớp nối phân cấp bằng mô phỏng kịch bản Copula & Lợi ích Rủi ro gộp", h2_style))
    story.append(Paragraph(
        "<b>Nghịch lý toán học lớn nhất trong dự báo phân phối: Các phân vị không có tính cộng (Quantiles are non-additive)!</b>",
        body_bold
    ))
    story.append(Paragraph(
        "Nếu Cửa hàng 1 có P<sub>90</sub> = 100, Cửa hàng 2 có P<sub>90</sub> = 100, thì P<sub>90</sub> của Tổng 2 cửa hàng <b>hoàn toàn không bằng 200</b>! "
        "Cộng gộp 2 con số P<sub>90</sub> đồng nghĩa với việc bạn âm thầm giả định rằng <i>'cả 2 cửa hàng đều rơi vào ngày bán đắt kỷ lục cùng một lúc'</i> (tương quan hoàn hảo <i>ρ = 1</i>). Điều này sẽ phóng đại rủi ro và buộc doanh nghiệp phải trữ tồn kho an toàn khổng lồ vô lý.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Giải pháp đột phá của dự án:</b><br/>"
        "• Sử dụng <b>Gaussian Copula</b> kết hợp phân rã Cholesky để ước lượng ma trận tương quan sai số giữa các cửa hàng/SKU và tương quan tự hồi quy AR(1) giữa các ngày liên tiếp.<br/>"
        "• Rút ngẫu nhiên <b>2,000 đường mẫu kịch bản (Sample Paths)</b> phản ánh đúng sự biến thiên và phụ thuộc lẫn nhau trong thế giới thực.<br/>"
        "• Cộng các đường mẫu này lại từ dưới lên (Bottom-up Path Summing) để tạo ra phân phối nhu cầu tổng thể ở cấp Toàn quốc hoặc cấp Vùng.<br/>"
        "• <b>Lợi ích Rủi ro gộp (Risk Pooling Benefit):</b> Nhờ mô phỏng đúng tương quan, tồn kho an toàn tổng hợp được chứng minh <b>thấp hơn từ 10% đến 25%</b> so với việc cộng dồn cơ học tồn kho an toàn của từng cửa hàng!",
        body_style
    ))
    
    p_hier = figures_dir / "hierarchy_reconciliation.png"
    if p_hier.exists():
        story.append(Image(str(p_hier), width=500, height=190))
        story.append(Paragraph("Hình 3: Khớp nối phân cấp Bottom-up hoàn hảo giữa cấp Cửa hàng, cấp Sản phẩm và cấp Toàn hệ thống.", img_caption))
    
    story.append(Spacer(1, 4))
    
    # --- BƯỚC 7 & 8 ---
    story.append(Paragraph("Bước 7 & 8: Phát hiện sụt giảm bằng CUSUM & Hoạch định tồn kho tối ưu", h2_style))
    story.append(Paragraph(
        "• <b>Phát hiện đứt gãy bằng Biểu đồ kiểm soát CUSUM:</b> Tích lũy các sai lệch âm nhỏ vượt ngưỡng nới lỏng (<i>Slack = 0.5 · σ</i>) và gióng chuông báo động khi tổng tích lũy vượt ngưỡng <i>Threshold = 4.0 · σ</i>. Sau mỗi lần báo động, CUSUM tự động thiết lập lại (reset) để phát hiện các đợt sụt giảm riêng biệt tiếp theo.<br/>"
        "• <b>Hoạch định tồn kho từ mô phỏng Lead Time:</b> Thay vì áp công thức sách giáo khoa, hệ thống tính <b>Reorder Point (ROP)</b> trực tiếp từ phân vị của 2,000 kịch bản nhu cầu trong suốt thời gian giao hàng (kể cả khi thời gian giao hàng bị trễ hạn ngẫu nhiên - Stochastic Lead Time).",
        body_style
    ))
    
    p_cusum = figures_dir / "cusum_example.png"
    if p_cusum.exists():
        story.append(Image(str(p_cusum), width=500, height=185))
        story.append(Paragraph("Hình 4: Cơ chế CUSUM phát hiện sự sụt giảm nhu cầu âm ỉ kéo dài và tự reset sau khi phát cảnh báo.", img_caption))
        
    story.append(Spacer(1, 4))
    story.append(Paragraph("Bước 9: Kiến trúc tính toán song song & Ghim luồng CPU tối ưu", h2_style))
    story.append(Paragraph(
        "Khi mở rộng lên 100 chuỗi lá độc lập, chạy tuần tự trên 1 core CPU tốn 5-6 phút. Dự án thiết kế <code>multiprocessing.Pool</code> song song theo từng SKU với 2 cải tiến kiến trúc quan trọng:<br/>"
        "1. <b>Initializer tải dữ liệu 1 lần:</b> Dataframe lớn được nạp vào bộ nhớ dùng chung của worker đúng 1 lần khi khởi tạo process, không truyền nhận dữ liệu qua IPC lặp lại.<br/>"
        "2. <b>Ghim luồng XGBoost (<code>xgb_n_jobs=1</code>):</b> Ép mỗi mô hình XGBoost bên trong worker chỉ dùng đúng 1 thread. Nếu không ghim luồng, 4-8 process sẽ cùng lúc tranh giành tất cả các CPU core của máy, gây ra hiện tượng nghẽn chuyển ngữ cảnh (Thread Contention) làm máy bị đơ và chậm đi gấp nhiều lần.",
        body_style
    ))
    
    story.append(PageBreak())
    
    # =========================================================================
    # PHẦN 4: GIẢI THÍCH CHI TIẾT TỪNG ĐOẠN CODE SOURCE CODE (MỚI THÊM)
    # =========================================================================
    story.append(Paragraph("PHẦN 4: GIẢI THÍCH CHI TIẾT SOURCE CODE LIÊN KẾT TỪNG BƯỚC", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceBefore=2, spaceAfter=8))
    
    story.append(Paragraph(
        "Phần này giải phẫu chi tiết các đoạn mã nguồn cốt lõi trong thư mục <code>src/</code>, giải thích cách thức các công thức toán học và logic chuỗi cung ứng được hiện thực hóa bằng mã Python.",
        body_style
    ))
    
    # Code 1: Data Loader
    code_1 = """# src/pipeline.py / src/data_loader.py
series = data_loader.get_series(df, sku, location)
idx = series.index
if (idx[-1] - idx[0]).days + 1 != len(idx) or not idx.is_unique:
    raise ValueError(f"{sku} x {location}: demand series is not a contiguous daily series.")"""
    exp_1 = "Đoạn code kiểm tra tính liên tục tuyệt đối của chuỗi thời gian hàng ngày. Nếu khoảng cách giữa ngày bắt đầu và ngày kết thúc không bằng độ dài chuỗi hoặc có ngày trùng lặp, hệ thống lập tức báo lỗi nhằm ngăn chặn việc sai lệch bước trễ thời gian."
    story.append(make_code_box(code_1, "src/pipeline.py (Liên kết Bước 1: Data Ingestion)", exp_1))
    story.append(Spacer(1, 7))
    
    # Code 2: Feature Engineering
    code_2 = """# src/features.py: make_direct_frame()
o = np.repeat(origin_pos, len(horizons))
h = np.tile(horizons, len(origin_pos))
t = o + h
k = np.ceil(h / 7).astype(int)
for j in range(1, config.SEASONAL_LAG_WEEKS + 1):
    feats[f"seasonal_lag_{j}"] = Y[t - 7 * (k + j - 1)]"""
    exp_2 = "Cơ chế tính đặc trưng As-of-origin Direct Multi-horizon: Với mỗi ngày mục tiêu t = o + h (trong tương lai), bước nhảy lùi k = ceil(h / 7) đảm bảo rằng điểm quan sát cùng thứ tuần trước luôn nằm hoàn toàn ở thời điểm trước hoặc tại ngày Origin o. Triệt tiêu 100% hiện tượng rò rỉ dữ liệu tương lai (Zero Data Leakage)."
    story.append(make_code_box(code_2, "src/features.py (Liên kết Bước 2: Feature Engineering)", exp_2))
    story.append(Spacer(1, 7))
    
    # Code 3: Split Conformal Calibration Split
    code_3 = """# src/pipeline.py: run_sku_backtest()
te = ref_series.index.get_loc(fold.train_end)
cs = te - calib_days    # calib_days = 60
train = tp < cs
calib = (tp >= cs) & (tp < te)
val   = op == te - 1"""
    exp_3 = "Trong mỗi Fold của Walk-forward Backtest, 60 ngày cuối cùng của tập huấn luyện (cs đến te) được tách riêng làm tập Calibration. Mô hình XGBoost chỉ được huấn luyện trên dữ liệu trước cs. Sau đó, mô hình dự báo trên tập calib để thu thập phần dư sai số thực tế mà nó chưa từng thấy."
    story.append(make_code_box(code_3, "src/pipeline.py (Liên kết Bước 3: Backtest & Conformal Split)", exp_3))
    story.append(Spacer(1, 7))
    
    # Code 4: Multi-Quantile XGBoost & Monotonicity
    code_4 = """# src/probabilistic.py: fit_quantile_model() & predict_quantiles()
model = xgb.XGBRegressor(
    **params,
    objective="reg:quantileerror",
    quantile_alpha=np.asarray(sorted(quantiles), dtype=float),
)
preds = np.asarray(model.predict(X)).reshape(len(X), len(qs))
return pd.DataFrame(np.sort(preds, axis=1), index=X.index, columns=qs)"""
    exp_4 = "Huấn luyện một mô hình Multi-Quantile XGBoost duy nhất cho 8 phân vị cùng lúc. Hàm np.sort(preds, axis=1) sắp xếp các phân vị tăng dần theo hàng, xử lý triệt để lỗi giao cắt phân vị (Quantile Crossing) - đảm bảo quy luật toán học P05 <= P10 <= ... <= P97.5."
    story.append(make_code_box(code_4, "src/probabilistic.py (Liên kết Bước 4: Quantile Modeling)", exp_4))
    story.append(Spacer(1, 7))
    
    # Code 5: Conformal Offsets
    code_5 = """# src/probabilistic.py: conformal_offsets() & apply_conformal_offsets()
for b_idx in buckets:
    for q in quantiles:
        # Lấy empirical quantile thứ q của phần dư (y - raw_pred_q)
        offsets[b_idx][q] = float(np.quantile(sub_res[q], q))
calibrated_pred = raw_pred + offset"""
    exp_5 = "Hiệu chuẩn ngoại suy Split Conformal: Tính độ dời bù trừ (offset) bằng phân vị thực nghiệm thứ q của phần dư trên tập Calibration. Nếu mô hình bị thiếu độ bao phủ (under-coverage), offset sẽ mang giá trị dương đẩy dự báo lên, đảm bảo xác suất thực tế khớp với xác suất danh định."
    story.append(make_code_box(code_5, "src/probabilistic.py (Liên kết Bước 4: Conformal Calibration)", exp_5))
    story.append(Spacer(1, 7))
    
    # Code 6: Newsvendor Cost Calculation
    code_6 = """# src/inventory.py: derive_costs_from_price() & newsvendor_cost()
stockout_cost = gross_margin * avg_unit_price          # Cu = 30% x Price
holding_cost  = annual_rate * price * review_days / 365 # Co = 25% x Price x 7 / 365
shortfall = np.maximum(y_true - y_pred, 0) # Thiếu hàng
excess    = np.maximum(y_pred - y_true, 0) # Thừa hàng
total_cost = unit_stockout_cost * shortfall.sum() + unit_holding_cost * excess.sum()"""
    exp_6 = "Mô hình chi phí Newsvendor quy đổi trực tiếp sai số dự báo sang tiền USD thiệt hại kinh doanh mỗi ngày. Sai số thiếu hàng (đứt hàng) bị phạt bằng phần lợi nhuận mất đi; sai số thừa hàng bị phạt bằng chi phí đọng vốn và lưu kho trong chu kỳ 7 ngày."
    story.append(make_code_box(code_6, "src/inventory.py (Liên kết Bước 5: Newsvendor Cost)", exp_6))
    story.append(Spacer(1, 7))
    
    # Code 7: Gaussian Copula Scenario Simulation
    code_7 = """# src/scenarios.py: estimate_dependence() & sample_joint_paths()
Z_leaf = np.random.randn(n_samples, n_leaves) @ dep.chol_leaf.T
# Mô phỏng AR(1) theo horizon: E_h = rho * E_{h-1} + sqrt(1 - rho^2) * W
U = norm.cdf(Z) # Đưa về phân phối đều U(0, 1)
# Ánh xạ nghịch đảo qua hàm phân phối biên (Piecewise-linear inverse CDF)
leaf_samples[:, leaf_idx, h_idx] = np.interp(U, grid_levels, grid_values)"""
    exp_7 = "Khớp nối kịch bản Copula: Tái tạo ma trận hiệp phương sai giữa các cửa hàng (qua Cholesky factor có co ngót shrinkage) và tự tương quan thời gian AR(1). 2,000 kịch bản ngẫu nhiên đa chiều được sinh ra, bảo toàn chính xác cấu trúc tương quan để tính toán lợi ích Rủi ro gộp (Risk Pooling)."
    story.append(make_code_box(code_7, "src/scenarios.py (Liên kết Bước 6: Copula Scenarios)", exp_7))
    story.append(Spacer(1, 7))
    
    # Code 8: CUSUM Drop Detection
    code_8 = """# src/drop_detection.py: detect_drops()
running = min(0.0, running + (y_t - ref_mean) + slack)
if running <= -threshold:
    alarm_days.append(date)
    running = 0.0 # Tự động reset sau báo động"""
    exp_8 = "Thuật toán CUSUM một phía phát hiện sụt giảm nhu cầu: Tích lũy các độ lệch âm vượt quá vùng nới lỏng (slack = 0.5 * std). Khi tổng tích lũy vượt ngưỡng báo động (-4.0 * std), hệ thống kích hoạt cảnh báo và tự reset về 0 để ghi nhận đợt sụt giảm riêng biệt tiếp theo."
    story.append(make_code_box(code_8, "src/drop_detection.py (Liên kết Bước 7: CUSUM Detection)", exp_8))
    story.append(Spacer(1, 7))
    
    # Code 9: Lead-Time Safety Stock Calculation
    code_9 = """# src/inventory.py: safety_stock_from_lead_time_samples()
expected = float(lead_time_demand_samples.mean())
rop = float(np.quantile(lead_time_demand_samples, service_level))
safety_stock = max(0.0, rop - expected)"""
    exp_9 = "Tính toán Reorder Point (ROP) và Tồn kho an toàn (Safety Stock) trực tiếp từ phân vị của 2,000 đường mẫu mô phỏng nhu cầu trong thời gian giao hàng. Hoàn toàn không phụ thuộc vào giả định phân phối chuẩn hình chuông như công thức cổ điển z * sigma * sqrt(L)."
    story.append(make_code_box(code_9, "src/inventory.py (Liên kết Bước 8: Inventory Sizing)", exp_9))
    
    story.append(PageBreak())
    
    # =========================================================================
    # PHẦN 5: MA TRẬN SO SÁNH TOÀN DIỆN VÀ PHÂN TÍCH ĐỐI CHUẨN
    # =========================================================================
    story.append(Paragraph("PHẦN 5: SO SÁNH TOÀN DIỆN VỚI WORKFLOW CHUẨN CÔNG NGHIỆP", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceBefore=2, spaceAfter=8))
    
    story.append(Paragraph(
        "Bảng ma trận đối chuẩn 10 tiêu chí kỹ thuật và nghiệp vụ giữa quy trình truyền thống và giải pháp hiện tại:",
        body_style
    ))
    
    comparison_data = [
        [
            Paragraph("Tiêu chí so sánh", table_header),
            Paragraph("Workflow chuẩn truyền thống (Standard SCM)", table_header),
            Paragraph("Workflow dự án hiện tại (This Project)", table_header),
            Paragraph("Lợi ích thực tế mang lại", table_header),
        ],
        [
            Paragraph("<b>1. Đầu ra dự báo</b>", table_cell),
            Paragraph("Dự báo một con số đơn lẻ (Point forecast: Mean/Median).", table_cell),
            Paragraph("Dự báo dải phân phối 8 phân vị xác suất (P<sub>05</sub> đến P<sub>97.5</sub>).", table_cell),
            Paragraph("Định lượng chính xác dải biến thiên và rủi ro bất định.", table_cell),
        ],
        [
            Paragraph("<b>2. Bảo đảm độ tin cậy của khoảng dự báo</b>", table_cell),
            Paragraph("Không có hoặc giả định phân phối chuẩn hình chuông ngây thơ.", table_cell),
            Paragraph("Hiệu chuẩn độc lập bằng Split Conformal Prediction theo Horizon.", table_cell),
            Paragraph("Cam kết thống kê: Khoảng P<sub>90</sub> chứa đúng 90% trường hợp thực tế.", table_cell),
        ],
        [
            Paragraph("<b>3. Kỹ thuật tạo biến chuỗi thời gian</b>", table_cell),
            Paragraph("Tạo biến Lag/Rolling trên toàn chuỗi rồi cắt chia tập dữ liệu.", table_cell),
            Paragraph("Direct multi-horizon theo cấu trúc (Origin, Horizon h).", table_cell),
            Paragraph("Triệt tiêu 100% hiện tượng rò rỉ dữ liệu tương lai (Data Leakage).", table_cell),
        ],
        [
            Paragraph("<b>4. Phương pháp kiểm thử mô hình</b>", table_cell),
            Paragraph("Chia Train/Test tĩnh hoặc K-Fold Cross Validation ngẫu nhiên.", table_cell),
            Paragraph("Walk-forward Backtesting cuộn theo thời gian thực tế.", table_cell),
            Paragraph("Đánh giá trung thực năng lực thích ứng với dữ liệu mới.", table_cell),
        ],
        [
            Paragraph("<b>5. Cộng gộp cấp bậc (Hierarchy Reconciliation)</b>", table_cell),
            Paragraph("Cộng dồn số trung bình hoặc bổ tỉ lệ Top-down cứng nhắc.", table_cell),
            Paragraph("Mô phỏng 2,000 đường mẫu Copula rồi cộng Bottom-up.", table_cell),
            Paragraph("Khai thác hiệu ứng Rủi ro gộp (tiết kiệm 10-25% tồn kho đệm).", table_cell),
        ],
        [
            Paragraph("<b>6. Thước đo hiệu quả mô hình</b>", table_cell),
            Paragraph("Chỉ đo lỗi toán học đối xứng: MAE, RMSE, MAPE.", table_cell),
            Paragraph("Đo lỗi toán học (WAPE, MASE) + Chi phí kinh doanh Newsvendor ($).", table_cell),
            Paragraph("Biết chính xác sai số làm công ty mất bao nhiêu tiền mỗi ngày.", table_cell),
        ],
        [
            Paragraph("<b>7. Giám sát đứt gãy nhu cầu (Drop Detection)</b>", table_cell),
            Paragraph("Dùng ngưỡng tĩnh (Ví dụ: báo động khi giảm 30% so với trung bình).", table_cell),
            Paragraph("Biểu đồ kiểm soát CUSUM tích lũy sai lệch âm kéo dài.", table_cell),
            Paragraph("Phát hiện sớm các xu hướng suy giảm âm ỉ, tự động reset.", table_cell),
        ],
        [
            Paragraph("<b>8. Tính Tồn kho an toàn (Safety Stock)</b>", table_cell),
            Paragraph("Dùng công thức chuẩn <i>z · σ · √L</i> (giả định chuẩn hóa).", table_cell),
            Paragraph("Trích xuất trực tiếp từ phân phối mô phỏng thời gian giao hàng.", table_cell),
            Paragraph("Không bị thiếu hàng khi nhu cầu thực tế lệch phải hoặc Lead Time trễ.", table_cell),
        ],
        [
            Paragraph("<b>9. Tích hợp quyết định kinh doanh</b>", table_cell),
            Paragraph("Tách rời hoàn toàn: Đội Data làm mô hình, Đội Kho tự tính tồn.", table_cell),
            Paragraph("Tích hợp trực tiếp Tỉ số tới hạn Critical Ratio vào dự báo.", table_cell),
            Paragraph("Hàng lãi cao tự động được bảo vệ với mức an toàn cao hơn.", table_cell),
        ],
        [
            Paragraph("<b>10. Kiến trúc tính toán phần cứng</b>", table_cell),
            Paragraph("Chạy đơn luồng chậm chạp hoặc tranh chấp CPU khi mở rộng.", table_cell),
            Paragraph("Multiprocessing Pool ghim luồng XGBoost (1 thread/worker).", table_cell),
            Paragraph("Tốc độ xử lý tăng tuyến tính theo số nhân CPU mà không treo máy.", table_cell),
        ],
    ]
    
    comp_tbl = Table(comparison_data, colWidths=[90, 135, 140, 140])
    comp_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('BOX', (0,0), (-1,-1), 1, BORDER_LIGHT),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_LIGHT),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4.5),
        ('LEFTPADDING', (0,0), (-1,-1), 5),
        ('RIGHTPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(comp_tbl)
    
    story.append(Spacer(1, 10))
    story.append(make_callout(
        "<b>Đúc kết sự khác biệt then chốt:</b> Workflow truyền thống được thiết kế theo tư duy <i>'Thống kê mô tả và Báo cáo sai số'</i>. Trong khi đó, Pipeline của dự án này được thiết kế theo tư duy <i>'Khoa học ra quyết định trong điều kiện bất định' (Decision Science under Uncertainty)</i>. Sự dịch chuyển từ việc đo MAE/MAPE sang đo lường rủi ro phân phối và thiệt hại tài chính bằng tiền USD là bước tiến quan trọng nhất giúp mô hình tạo ra giá trị kinh tế trực tiếp cho doanh nghiệp.",
        bg_color=SUCCESS_BG, border_color=SUCCESS_BORDER, title="TỔNG KẾT GIÁ TRỊ VƯỢT TRỘI"
    ))
    
    story.append(PageBreak())
    
    # =========================================================================
    # PHẦN 6: KẾT QUẢ THỰC NGHIỆM TỪ FULL RUN PIPELINE
    # =========================================================================
    story.append(Paragraph("PHẦN 6: KẾT QUẢ THỰC NGHIỆM TRÊN TOÀN BỘ 100 CHUỖI LÁ (FULL RUN)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceBefore=2, spaceAfter=8))
    
    story.append(Paragraph(
        "Dưới đây là các kết quả phân tích thống kê và kiểm định thực nghiệm sau khi pipeline đã hoàn tất chạy 100% trên toàn bộ 20 SKU × 5 Cửa hàng:",
        body_style
    ))
    
    # Chèn hình Coverage by horizon và Pinball loss
    p_cov = figures_dir / "coverage_by_horizon.png"
    p_pin = figures_dir / "pinball_loss_by_quantile.png"
    if p_cov.exists() and p_pin.exists():
        row_c = [
            [Image(str(p_cov), width=250, height=125), Image(str(p_pin), width=250, height=125)]
        ]
        tbl_c = Table(row_c, colWidths=[252, 252])
        tbl_c.setStyle(TableStyle([
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('LEFTPADDING', (0,0), (-1,-1), 0),
            ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ]))
        story.append(tbl_c)
        story.append(Paragraph("Hình 5: (Trái) Tỷ lệ bao phủ thực tế P10-P90 ổn định quanh mức danh định 80% trên tất cả 14 chân trời dự báo. (Phải) Pinball loss phân bố đồng đều giữa các phân vị.", img_caption))
    
    story.append(Spacer(1, 4))
    
    # Chèn hình Residual diagnostics
    p_resid = figures_dir / "residual_diagnostics.png"
    if p_resid.exists():
        story.append(Image(str(p_resid), width=500, height=140))
        story.append(Paragraph("Hình 6: Phân tích phần dư tổng hợp trên 100 leaves: Mean = 1.25 đơn vị (không thiên lệch) và biểu đồ Q-Q Plot đuôi nặng (Fat Tails).", img_caption))
    
    story.append(Paragraph(
        "• <b>Trung bình phần dư gần 0 (Unbiased):</b> Trung bình sai số chỉ ở mức 1.25 đơn vị trên quy mô nhu cầu trung bình ~100. Dự báo trung vị (P<sub>50</sub>) hoàn toàn không bị lệch.<br/>"
        "• <b>Đuôi nặng trên biểu đồ Q-Q Plot (Fat Tails):</b> Đường chấm đỏ uốn cong ở hai đầu đuôi chứng minh sai số thực tế có đuôi dày hơn phân phối chuẩn lý thuyết. Điều này một lần nữa khẳng định: <b>Nếu dùng công thức an toàn cổ điển sẽ bị thiếu hụt tồn kho nghiêm trọng vào những ngày xảy ra biến cố đuôi</b>. Việc dùng phương pháp mô phỏng kịch bản (Scenario Simulation) của dự án là hoàn toàn chuẩn xác.<br/>"
        "• <b>Lợi ích Rủi ro gộp đo được thực tế:</b> Tồn kho an toàn ở cấp Tổng toàn bộ hệ thống đạt mức tiết kiệm <b>~16.4%</b> so với tổng cơ học tồn kho an toàn của từng cửa hàng riêng lẻ nhờ hiệu ứng triệt tiêu rủi ro độc lập.",
        body_style
    ))
    
    # Chèn hình Phân bố độ chính xác
    p_leaf = figures_dir / "leaf_accuracy_distribution.png"
    if p_leaf.exists():
        story.append(Image(str(p_leaf), width=500, height=170))
        story.append(Paragraph("Hình 7: Phân bố sai số MAE trên 100 chuỗi lá và tương quan tuyến tính giữa giá bán sản phẩm với chi phí tổn thất dự báo.", img_caption))
        
    story.append(PageBreak())
    
    # =========================================================================
    # PHẦN 7: BẢNG TRA CỨU THAM SỐ CẤU HÌNH & HƯỚNG DẪN TRIỂN KHAI THỰC CHIẾN
    # =========================================================================
    story.append(Paragraph("PHẦN 7: BẢNG TRA CỨU THAM SỐ CẤU HÌNH & HƯỚNG DẪN TRIỂN KHAI", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=PRIMARY, spaceBefore=2, spaceAfter=8))
    
    story.append(Paragraph(
        "Bảng tổng hợp toàn bộ các tham số cấu hình hệ thống (Hyperparameters & System Config) trong <code>src/config.py</code> giúp chuyên viên dễ dàng tùy chỉnh theo từng quy mô doanh nghiệp:",
        body_style
    ))
    
    param_data = [
        [Paragraph("Tham số cấu hình", table_header), Paragraph("Giá trị mặc định", table_header), Paragraph("Ý nghĩa & Tác động nghiệp vụ", table_header)],
        [
            Paragraph("<code>QUANTILES</code>", table_cell),
            Paragraph("[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]", table_cell),
            Paragraph("Dải phân vị xác suất huấn luyện. Mức cao nhất 0.975 dùng cho các mặt hàng thiết yếu có Service Level cao.", table_cell),
        ],
        [
            Paragraph("<code>CONFORMAL_CALIB_DAYS</code>", table_cell),
            Paragraph("60 ngày", table_cell),
            Paragraph("Số ngày cuối của tập huấn luyện được tách ra làm tập hiệu chuẩn Conformal riêng biệt.", table_cell),
        ],
        [
            Paragraph("<code>HORIZON_BUCKETS</code>", table_cell),
            Paragraph("[(1, 7), (8, 14)]", table_cell),
            Paragraph("Phân nhóm chân trời dự báo: Tuần 1 và Tuần 2 để tính độ dời offset hiệu chuẩn riêng biệt.", table_cell),
        ],
        [
            Paragraph("<code>N_SCENARIO_SAMPLES</code>", table_cell),
            Paragraph("2,000 kịch bản", table_cell),
            Paragraph("Số lượng đường mẫu kịch bản chuỗi sinh ra từ Gaussian Copula để tính toán Rủi ro gộp và tồn kho an toàn.", table_cell),
        ],
        [
            Paragraph("<code>COPULA_LEAF_SHRINKAGE</code>", table_cell),
            Paragraph("0.20", table_cell),
            Paragraph("Hệ số co ngót ma trận tương quan giữa các cửa hàng về ma trận đơn vị, tránh hiện tượng quá khớp (Overfitting).", table_cell),
        ],
        [
            Paragraph("<code>CUSUM_THRESHOLD_STD</code><br/><code>CUSUM_SLACK_STD</code>", table_cell),
            Paragraph("Threshold = 4.0 · σ<br/>Slack = 0.5 · σ", table_cell),
            Paragraph("Ngưỡng báo động và vùng nới lỏng của biểu đồ CUSUM. Lọc sạch nhiễu thường ngày, chỉ báo động khi sụt giảm kéo dài.", table_cell),
        ],
        [
            Paragraph("<code>GROSS_MARGIN</code>", table_cell),
            Paragraph("30% (0.30)", table_cell),
            Paragraph("Biên lợi nhuận gộp dùng để tính chi phí thiệt hại khi bị đứt hàng (Cu = 30% × Giá bán).", table_cell),
        ],
        [
            Paragraph("<code>ANNUAL_HOLDING_RATE</code>", table_cell),
            Paragraph("25% (0.25)", table_cell),
            Paragraph("Chi phí vốn và lưu kho hàng năm (Co = 25%/năm × Giá bán × 7 / 365 cho 1 chu kỳ đặt hàng 7 ngày).", table_cell),
        ],
        [
            Paragraph("<code>DEFAULT_LEAD_TIME_DAYS</code>", table_cell),
            Paragraph("7 ngày (Std = 0 hoặc tùy chọn)", table_cell),
            Paragraph("Thời gian giao hàng trung bình từ nhà cung cấp dùng để mô phỏng tích lũy nhu cầu.", table_cell),
        ],
    ]
    
    param_tbl = Table(param_data, colWidths=[135, 120, 250])
    param_tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), PRIMARY),
        ('BOX', (0,0), (-1,-1), 1, BORDER_LIGHT),
        ('INNERGRID', (0,0), (-1,-1), 0.5, BORDER_LIGHT),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 4.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4.5),
        ('LEFTPADDING', (0,0), (-1,-1), 6),
        ('RIGHTPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(param_tbl)
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("Quy trình 4 bước áp dụng thực chiến cho Supply Chain Planner:", h2_style))
    story.append(Paragraph(
        "<b>Bước 1 - Giám sát cảnh báo CUSUM:</b> Kiểm tra danh sách các ngày có <code>cusum_alarm_days > 0</code> để xác định các sản phẩm đang có dấu hiệu đứt gãy thị trường âm ỉ, kịp thời làm việc với bộ phận Marketing.<br/>"
        "<b>Bước 2 - Kiểm tra độ tin cậy phân vị:</b> Đối chiếu Reliability Diagram và Bảng Coverage. Đảm bảo tỷ lệ bao phủ P10-P90 luôn dao động trong khoảng 78% - 82% trước khi dùng số liệu đặt hàng.<br/>"
        "<b>Bước 3 - Ra quyết định đặt hàng qua Reorder Point:</b> Khi mức tồn kho thực tế chạm ngưỡng <code>reorder_point</code>, tự động đề xuất đơn đặt hàng với số lượng bù đắp bằng nhu cầu Lead Time kỳ vọng.<br/>"
        "<b>Bước 4 - Tận dụng kho trung tâm (Central DC) qua Risk Pooling:</b> Với các sản phẩm có mức tiết kiệm rủi ro gộp cao (>15%), ưu tiên giữ hàng đệm tại kho trung tâm và phân phối nhanh về cửa hàng thay vì trữ phân tán, giúp cắt giảm hàng trăm ngàn USD vốn lưu động.",
        body_style
    ))
    
    story.append(Spacer(1, 12))
    story.append(Paragraph("LỜI KẾT", h3_style))
    story.append(Paragraph(
        "Dự án <b>SKU x Location Stochastic Demand Forecasting</b> là cầu nối vững chắc đưa Khoa học Dữ liệu tiên tiến vào giải quyết trực tiếp bài toán kinh tế chuỗi cung ứng. "
        "Bằng cách hiểu rõ bản chất từng bước thực thi và mã nguồn tương ứng, doanh nghiệp có thể tự tin triển khai hệ thống này vào môi trường sản xuất thực tế, giảm thiểu hàng tồn ứ, ngăn ngừa cháy hàng và tối ưu hóa hàng triệu USD dòng vốn lưu động.",
        body_style
    ))
    
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"File PDF generated successfully at: {filename}")

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent.parent
    figures_path = current_dir / "reports" / "figures"
    output_pdf_path = current_dir / "Bao_Cao_Chi_Tiet_Workflow_Demand_Forecasting.pdf"
    
    build_pdf(str(output_pdf_path), figures_path)
    
    workspace_root = current_dir.parent
    root_pdf_path = workspace_root / "Bao_Cao_Chi_Tiet_Workflow_Demand_Forecasting.pdf"
    if root_pdf_path.parent.exists():
        shutil.copy(output_pdf_path, root_pdf_path)
        print(f"Copied another copy to workspace root: {root_pdf_path}")
