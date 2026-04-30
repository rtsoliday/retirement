package com.retirementreadinesslab.reports

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Typeface
import android.graphics.pdf.PdfDocument
import com.retirementreadinesslab.model.RetirementScenario
import com.retirementreadinesslab.model.SimulationResult
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

object ReportPdfExporter {
    private const val PAGE_WIDTH = 612
    private const val PAGE_HEIGHT = 792
    private const val PAGE_MARGIN = 42f
    private const val LINE_HEIGHT = 16f

    fun createReportPdf(
        context: Context,
        scenario: RetirementScenario,
        result: SimulationResult?
    ): File {
        val reportsDir = File(context.cacheDir, "shared_reports").apply { mkdirs() }
        val file = File(reportsDir, "retirement-readiness-${safeFileName(scenario.name)}.pdf")
        val document = PdfDocument()

        try {
            val reportLines = ReportBuilder.buildTextReport(scenario, result)
                .replace("•", "-")
                .lineSequence()
                .flatMap { line -> wrapLine(line, maxChars = 86) }
                .toList()
            writeLines(document, reportLines)
            file.outputStream().use { output ->
                document.writeTo(output)
            }
        } finally {
            document.close()
        }

        return file
    }

    private fun writeLines(document: PdfDocument, lines: List<String>) {
        val titlePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFF17201C.toInt()
            textSize = 21f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }
        val headingPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFF176B5B.toInt()
            textSize = 14f
            typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD)
        }
        val bodyPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFF17201C.toInt()
            textSize = 10.5f
        }
        val mutedPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFF65706A.toInt()
            textSize = 9.5f
        }
        val dividerPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = 0xFFDDE3DE.toInt()
            strokeWidth = 1f
        }

        var pageNumber = 0
        var page = newPage(document, ++pageNumber)
        var canvas = page.canvas
        var y = drawPageHeader(canvas, titlePaint, mutedPaint, dividerPaint, pageNumber)

        lines.forEach { line ->
            val extraGap = when {
                line.isBlank() -> LINE_HEIGHT * 0.55f
                isHeading(line) -> LINE_HEIGHT * 1.2f
                else -> LINE_HEIGHT
            }
            if (y + extraGap > PAGE_HEIGHT - PAGE_MARGIN) {
                document.finishPage(page)
                page = newPage(document, ++pageNumber)
                canvas = page.canvas
                y = drawPageHeader(canvas, titlePaint, mutedPaint, dividerPaint, pageNumber)
            }

            when {
                line.isBlank() -> y += LINE_HEIGHT * 0.55f
                isHeading(line) -> {
                    y += 8f
                    canvas.drawText(line, PAGE_MARGIN, y, headingPaint)
                    y += LINE_HEIGHT * 0.35f
                }
                else -> {
                    canvas.drawText(line, PAGE_MARGIN, y, bodyPaint)
                    y += LINE_HEIGHT
                }
            }
        }

        document.finishPage(page)
    }

    private fun newPage(document: PdfDocument, pageNumber: Int): PdfDocument.Page {
        val info = PdfDocument.PageInfo.Builder(PAGE_WIDTH, PAGE_HEIGHT, pageNumber).create()
        return document.startPage(info)
    }

    private fun drawPageHeader(
        canvas: Canvas,
        titlePaint: Paint,
        mutedPaint: Paint,
        dividerPaint: Paint,
        pageNumber: Int
    ): Float {
        canvas.drawText("Retirement Readiness Lab", PAGE_MARGIN, 46f, titlePaint)
        canvas.drawText("Educational retirement stress-test report", PAGE_MARGIN, 65f, mutedPaint)
        canvas.drawText("Page $pageNumber", PAGE_WIDTH - PAGE_MARGIN - 34f, 65f, mutedPaint)
        canvas.drawLine(PAGE_MARGIN, 78f, PAGE_WIDTH - PAGE_MARGIN, 78f, dividerPaint)
        return 104f
    }

    private fun wrapLine(line: String, maxChars: Int): Sequence<String> = sequence {
        if (line.length <= maxChars) {
            yield(line)
            return@sequence
        }

        var remaining = line
        while (remaining.length > maxChars) {
            val breakAt = remaining
                .take(maxChars + 1)
                .lastIndexOf(' ')
                .takeIf { it > 0 }
                ?: maxChars
            yield(remaining.take(breakAt).trimEnd())
            remaining = remaining.drop(breakAt).trimStart()
        }
        yield(remaining)
    }

    private fun isHeading(line: String): Boolean {
        return line in setOf("Summary", "Key assumptions", "Suggested next test", "Disclaimer")
    }

    private fun safeFileName(value: String): String {
        val slug = value
            .lowercase(Locale.US)
            .replace(Regex("[^a-z0-9]+"), "-")
            .trim('-')
            .ifBlank { "scenario" }
        val date = SimpleDateFormat("yyyyMMdd-HHmm", Locale.US).format(Date())
        return "$slug-$date"
    }
}
