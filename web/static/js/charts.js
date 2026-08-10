/**
 * İşçi Takip Sistemi - Chart.js Grafik Yapılandırma Modülü
 */

const ChartTheme = {
    getColors() {
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        return {
            grid: isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(0, 0, 0, 0.08)',
            text: isDark ? '#94A3B8' : '#475569',
            green: isDark ? '#10B981' : '#059669',
            greenFill: isDark ? 'rgba(16, 185, 129, 0.15)' : 'rgba(5, 150, 105, 0.15)',
            red: isDark ? '#EF4444' : '#DC2626',
            redFill: isDark ? 'rgba(239, 68, 68, 0.15)' : 'rgba(220, 38, 38, 0.15)',
            accent: isDark ? '#3B82F6' : '#2563EB'
        };
    }
};

// --- AKTİVİTE ÇİZGİ GRAFİĞİ ---
const ActivityChart = {
    chart: null,
    
    init(canvasId = 'activityChart') {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;
        
        const colors = ChartTheme.getColors();
        
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Aktif Süre',
                        data: [],
                        borderColor: colors.green,
                        backgroundColor: colors.greenFill,
                        fill: true,
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 3
                    },
                    {
                        label: 'İnaktif Süre',
                        data: [],
                        borderColor: colors.red,
                        backgroundColor: colors.redFill,
                        fill: true,
                        tension: 0.4,
                        borderWidth: 2,
                        pointRadius: 3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: { color: colors.text, font: { family: 'Inter', size: 12 } }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false
                    }
                },
                scales: {
                    x: {
                        grid: { color: colors.grid },
                        ticks: { color: colors.text }
                    },
                    y: {
                        grid: { color: colors.grid },
                        ticks: { color: colors.text },
                        beginAtZero: true
                    }
                }
            }
        });
        
        this.loadData();
    },
    
    async loadData() {
        if (!this.chart) return;
        try {
            const res = await API.get('/api/reports/chart_data');
            if (res && res.labels) {
                this.chart.data.labels = res.labels;
                this.chart.data.datasets[0].data = res.aktif;
                this.chart.data.datasets[1].data = res.inaktif;
                this.chart.update();
            }
        } catch (e) {
            console.error('Aktivite grafik verisi çekilemedi:', e);
        }
    }
};

// --- ÖZET HALKA GRAFİĞİ ---
const SummaryChart = {
    chart: null,
    
    init(canvasId = 'summaryChart') {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;
        
        const colors = ChartTheme.getColors();
        
        this.chart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Aktif', 'İnaktif'],
                datasets: [{
                    data: [0, 0],
                    backgroundColor: [colors.green, colors.red],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: colors.text, font: { family: 'Inter', size: 12 } }
                    }
                }
            }
        });
    },
    
    update(aktif, inaktif) {
        if (this.chart) {
            this.chart.data.datasets[0].data = [aktif, inaktif];
            this.chart.update();
        }
    }
};

// Client-side CSV İndirme
function downloadCSV(data, filename = 'raporlar.csv') {
    if (!data || !data.length) {
        Toast.warning('İndirilecek veri bulunamadı.');
        return;
    }
    
    const headers = Object.keys(data[0]);
    const csvRows = [headers.join(',')];
    
    for (const row of data) {
        const values = headers.map(header => {
            const val = row[header] === null || row[header] === undefined ? '' : row[header];
            return `"${String(val).replace(/"/g, '""')}"`;
        });
        csvRows.push(values.join(','));
    }
    
    const blob = new Blob(['\ufeff' + csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

document.addEventListener('DOMContentLoaded', () => {
    ActivityChart.init();
    SummaryChart.init();
});
