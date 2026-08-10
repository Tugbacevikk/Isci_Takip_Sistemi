/**
 * İşçi Takip Sistemi - Ana JavaScript Modülü
 * Tema yönetimi, Socket.IO istemcisi, API istekleri ve dinamik Arayüz güncellemeleri
 */

// --- TEMA YÖNETİMİ (Sabit Beyaz Tema) ---
const ThemeManager = {
    init() {
        this.set('light');
    },
    
    set(theme) {
        document.documentElement.setAttribute('data-theme', 'light');
        localStorage.setItem('theme', 'light');
    },
    
    toggle() {
        this.set('light');
    }
};

// --- TOAST BİLDİRİM SİSTEMİ ---
const Toast = {
    container: null,
    
    init() {
        if (!this.container) {
            this.container = document.createElement('div');
            this.container.className = 'toast-container';
            document.body.appendChild(this.container);
        }
    },
    
    show(message, type = 'info', duration = 3000) {
        this.init();
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        let iconClass = 'fa-circle-info';
        if (type === 'success') iconClass = 'fa-circle-check';
        if (type === 'error') iconClass = 'fa-circle-xmark';
        if (type === 'warning') iconClass = 'fa-triangle-exclamation';
        
        toast.innerHTML = `
            <i class="fa-solid ${iconClass}"></i>
            <span>${message}</span>
        `;
        
        this.container.appendChild(toast);
        
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            toast.style.transition = 'all 0.3s ease-out';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    },
    
    success(msg) { this.show(msg, 'success'); },
    error(msg) { this.show(msg, 'error'); },
    warning(msg) { this.show(msg, 'warning'); },
    info(msg) { this.show(msg, 'info'); }
};

// Global erişim
window.showToast = (msg, type) => Toast.show(msg, type);

// --- API İSTEK YARDIMCISI ---
const API = {
    async get(url) {
        try {
            const res = await fetch(url);
            if (!res.ok) throw new Error(`HTTP Hata: ${res.status}`);
            return await res.json();
        } catch (e) {
            console.error(`API GET Hata (${url}):`, e);
            throw e;
        }
    },
    
    async post(url, data = {}) {
        try {
            const res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            if (!res.ok) throw new Error(`HTTP Hata: ${res.status}`);
            return await res.json();
        } catch (e) {
            console.error(`API POST Hata (${url}):`, e);
            throw e;
        }
    },

    async postForm(url, formData) {
        try {
            const res = await fetch(url, {
                method: 'POST',
                body: formData
            });
            if (!res.ok) throw new Error(`HTTP Hata: ${res.status}`);
            return await res.json();
        } catch (e) {
            console.error(`API Form POST Hata (${url}):`, e);
            throw e;
        }
    }
};

// --- REALTIME SOCKET.IO BAĞLANTISI ---
let socket = null;

let disconnectTimer = null;

const SocketManager = {
    init() {
        if (typeof io === 'undefined') return;
        
        socket = io();
        
        socket.on('connect', () => {
            if (disconnectTimer) {
                clearTimeout(disconnectTimer);
                disconnectTimer = null;
            }
            console.log('WebSocket Bağlantısı Başarılı');
            this.updateConnBadge(true);
        });
        
        socket.on('disconnect', () => {
            console.warn('WebSocket Bağlantısı Koptu');
            if (!disconnectTimer) {
                disconnectTimer = setTimeout(() => {
                    this.updateConnBadge(false);
                }, 3000);
            }
        });

        
        socket.on('status_update', (data) => {
            this.handleStatusUpdate(data);
        });

        socket.on('new_alarm', (data) => {
            Toast.warning(`ALARM: ${data.istasyon_adi} - ${data.aciklama}`);
            this.updateAlarmCount();
        });
    },
    
    updateConnBadge(connected) {
        const badge = document.getElementById('system-status-badge');
        if (badge) {
            badge.innerHTML = connected 
                ? '<span class="status-badge aktif"><span class="pulse-dot"></span> Bağlı</span>'
                : '<span class="status-badge yok">Bağlantı Kesildi</span>';
        }
    },
    
    handleStatusUpdate(data) {
        // UI Güncellemeleri
        const statusCircle = document.getElementById('status-circle');
        const statusTextBig = document.getElementById('status-text-big');
        const currentWorker = document.getElementById('current-worker');
        const liveStatusText = document.getElementById('live-status-text');
        const workerInfo = document.getElementById('worker-info');
        const kpiFps = document.getElementById('kpi-fps');
        const kpiCalisan = document.getElementById('kpi-calisan');
        const kpiDurum = document.getElementById('kpi-durum');
        
        let statusClass = 'yok';
        if (data.durum && data.durum.includes('AKTİF')) statusClass = 'aktif';
        if (data.durum && data.durum.includes('İNAKTİF')) statusClass = 'inaktif';
        if (data.durum && data.durum.includes('Telefon')) statusClass = 'telefon';
        
        if (statusCircle) {
            statusCircle.className = `status-circle ${statusClass}`;
        }
        if (statusTextBig) {
            statusTextBig.textContent = data.durum || 'Sistem Hazır';
        }
        if (currentWorker) {
            if (data.worker_name && data.worker_name.trim() !== '') {
                const cleanName = data.worker_name.replace(/\s*\([^)]*\)/g, '').trim();
                currentWorker.innerHTML = `<i class="fa-solid fa-user-tie"></i> ${cleanName}`;
            } else {
                currentWorker.innerHTML = `<i class="fa-solid fa-user-tie"></i> Çalışan tespit edilmedi`;
            }
        }
        if (liveStatusText) {
            liveStatusText.textContent = data.durum || 'Kamera Durduruldu';
        }
        if (workerInfo) {
            workerInfo.textContent = data.worker_name ? `Çalışan: ${data.worker_name}` : 'Çalışan: Tanımlanamadı';
        }
        if (kpiFps) {
            kpiFps.textContent = data.fps ? data.fps.toFixed(1) : '0';
        }
        if (kpiCalisan) {
            kpiCalisan.textContent = data.kisi_sayisi || '0';
        }
        if (kpiDurum) {
            kpiDurum.textContent = data.running ? 'Çalışıyor' : 'Durduruldu';
        }
    },
    
    async updateAlarmCount() {
        try {
            const res = await API.get('/api/alarms/unread_count');
            const badge = document.getElementById('alarm-count-badge');
            if (badge) {
                if (res.count > 0) {
                    badge.textContent = res.count;
                    badge.style.display = 'inline-block';
                } else {
                    badge.style.display = 'none';
                }
            }
        } catch (e) {}
    }
};

// --- CANLI SAAT GÜNCELLEMESİ ---
function updateClock() {
    const clockEl = document.getElementById('current-time');
    if (clockEl) {
        const now = new Date();
        clockEl.textContent = now.toLocaleTimeString('tr-TR');
    }
}

// --- DOM YÜKLENDİĞİNDE BAŞLAT ---
document.addEventListener('DOMContentLoaded', () => {
    ThemeManager.init();
    SocketManager.init();
    
    setInterval(updateClock, 1000);
    updateClock();
    
    SocketManager.updateAlarmCount();
    setInterval(() => SocketManager.updateAlarmCount(), 30000);
});
