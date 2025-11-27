"""
Simulador para Paper Académico - Arbitraje CEX-L2
Usa datos históricos para validar el modelo estocástico
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict


@dataclass
class HistoricalTick:
    """Tick histórico de precio"""
    timestamp: datetime
    cex_bid: float
    cex_ask: float
    l2_price: float
    l2_liquidity: float
    gas_price: float


@dataclass
class SimulatedTrade:
    """Trade simulado para análisis"""
    timestamp: datetime
    direction: str
    entry_cex: float
    entry_l2: float
    size: float
    gross_spread: float
    slippage: float
    gas_cost: float
    priority_fee: float
    net_spread: float
    p_inclusion: float
    pnl: float
    executed: bool  # Si realmente se ejecutó (basado en P_inclusion)


class DataSimulator:
    """Genera datos sintéticos realistas para simulación"""
    
    def __init__(self, days: int = 30, ticks_per_day: int = 86400):
        self.days = days
        self.ticks_per_day = ticks_per_day
        
    def generate_correlated_prices(self) -> pd.DataFrame:
        """
        Genera precios CEX y L2 correlacionados con ruido realista
        """
        n_ticks = self.days * self.ticks_per_day
        timestamps = pd.date_range(
            start='2024-01-01', 
            periods=n_ticks, 
            freq='1s'
        )
        
        # Precio base ETH con random walk + tendencia
        base_price = 2000
        returns = np.random.normal(0, 0.02/np.sqrt(86400), n_ticks)  # Vol anualizada 20%
        price_series = base_price * np.exp(np.cumsum(returns))
        
        # CEX: bid/ask spread 0.01-0.05%
        cex_spread = np.random.uniform(0.0001, 0.0005, n_ticks)
        cex_bid = price_series * (1 - cex_spread/2)
        cex_ask = price_series * (1 + cex_spread/2)
        
        # L2: precio con lag y ruido adicional (simulando liquidez fragmentada)
        l2_noise = np.random.normal(0, 0.001, n_ticks)  # ±0.1% ruido
        l2_lag = np.roll(price_series, 2)  # 2 segundos de lag
        l2_price = l2_lag * (1 + l2_noise)
        
        # Liquidez L2 (varía con hora del día)
        hour = timestamps.hour
        liquidity_multiplier = 1 + 0.5 * np.sin(2 * np.pi * hour / 24)  # Mayor en horas pico
        l2_liquidity = 500000 * liquidity_multiplier
        
        # Gas price (correlacionado con actividad)
        base_gas = 1.0  # gwei
        gas_spikes = np.random.exponential(2, n_ticks)
        gas_price = base_gas * gas_spikes
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'cex_bid': cex_bid,
            'cex_ask': cex_ask,
            'l2_price': l2_price,
            'l2_liquidity': l2_liquidity,
            'gas_price': gas_price
        })
        
        return df


class BacktestEngine:
    """Motor de backtesting con modelo estocástico"""
    
    def __init__(
        self,
        data: pd.DataFrame,
        min_spread: float = 0.002,
        min_p_inclusion: float = 0.7,
        trade_size: float = 1.0,
        initial_capital: float = 100000
    ):
        self.data = data
        self.min_spread = min_spread
        self.min_p_inclusion = min_p_inclusion
        self.trade_size = trade_size
        self.capital = initial_capital
        
        self.trades: List[SimulatedTrade] = []
        self.equity_curve = [initial_capital]
        
        # Modelo ACD para simulación
        self.acd_omega = 0.1
        self.acd_alpha = 0.3
        self.acd_beta = 0.6
        self.psi = 1.0
        self.last_trade_time = None
        
    def calculate_slippage(self, size: float, liquidity: float) -> float:
        """Modelo de slippage basado en liquidez"""
        return np.sqrt(1 + 2 * size * 2000 / liquidity) - 1  # Asume ETH @ $2000
    
    def update_acd_model(self, current_time: datetime):
        """Actualiza modelo ACD con duración desde último trade"""
        if self.last_trade_time is not None:
            duration = (current_time - self.last_trade_time).total_seconds()
            self.psi = self.acd_omega + self.acd_alpha * duration + self.acd_beta * self.psi
        self.last_trade_time = current_time
    
    def calculate_p_inclusion(self, priority_fee: float, gas_price: float) -> float:
        """
        Calcula P_inclusión basada en:
        - Tasa de llegada (lambda = 1/psi del modelo ACD)
        - Priority fee relativo al gas base
        """
        lambda_rate = 1.0 / max(self.psi, 0.1)
        expected_txs = lambda_rate * 2.0  # Block time 2s en L2
        
        # Percentil de fee
        fee_percentile = min(priority_fee / (gas_price * 2), 1.0)
        
        # Modelo: P = fee_percentile * (1 - congestión)
        congestion = min(expected_txs / 100, 0.8)  # Max 80% congestión
        p_inclusion = fee_percentile * (1 - congestion * 0.5)
        
        return np.clip(p_inclusion, 0.0, 1.0)
    
    def simulate_execution(self, p_inclusion: float) -> bool:
        """Simula si el trade se ejecuta basado en P_inclusión"""
        return np.random.random() < p_inclusion
    
    def run_backtest(self) -> Dict:
        """Ejecuta backtesting completo"""
        print(f"Iniciando backtest con {len(self.data)} ticks...")
        
        for idx, row in self.data.iterrows():
            # Calcula spread y dirección
            l2_to_cex_spread = (row['cex_bid'] - row['l2_price']) / row['l2_price']
            cex_to_l2_spread = (row['l2_price'] - row['cex_ask']) / row['cex_ask']
            
            best_spread = max(l2_to_cex_spread, cex_to_l2_spread)
            direction = "L2_TO_CEX" if l2_to_cex_spread > cex_to_l2_spread else "CEX_TO_L2"
            
            if best_spread < self.min_spread:
                continue
            
            # Calcula costos
            slippage = self.calculate_slippage(self.trade_size, row['l2_liquidity'])
            priority_fee_gwei = row['gas_price'] * 1.5  # 50% por encima de base
            gas_cost_usd = priority_fee_gwei * 300000 * 1e-9 * row['cex_bid']  # 300k gas
            priority_fee_pct = gas_cost_usd / (self.trade_size * row['l2_price'])
            
            cex_fee = 0.001  # 0.1% taker
            net_spread = best_spread - slippage - priority_fee_pct - cex_fee
            
            if net_spread < 0:
                continue
            
            # Actualiza modelo ACD y calcula P_inclusión
            self.update_acd_model(row['timestamp'])
            p_inclusion = self.calculate_p_inclusion(priority_fee_gwei, row['gas_price'])
            
            if p_inclusion < self.min_p_inclusion:
                continue
            
            # Simula ejecución
            executed = self.simulate_execution(p_inclusion)
            
            # Calcula PnL
            if executed:
                pnl = net_spread * self.trade_size * row['l2_price']
                self.capital += pnl
            else:
                pnl = -gas_cost_usd  # Solo perdemos gas si falla
                self.capital += pnl
            
            # Registra trade
            trade = SimulatedTrade(
                timestamp=row['timestamp'],
                direction=direction,
                entry_cex=row['cex_bid'] if direction == "L2_TO_CEX" else row['cex_ask'],
                entry_l2=row['l2_price'],
                size=self.trade_size,
                gross_spread=best_spread,
                slippage=slippage,
                gas_cost=gas_cost_usd,
                priority_fee=priority_fee_pct,
                net_spread=net_spread,
                p_inclusion=p_inclusion,
                pnl=pnl,
                executed=executed
            )
            self.trades.append(trade)
            self.equity_curve.append(self.capital)
            
            # Progress
            if idx % 10000 == 0:
                print(f"  Procesado: {idx}/{len(self.data)} ticks | Trades: {len(self.trades)}")
        
        return self._generate_statistics()
    
    def _generate_statistics(self) -> Dict:
        """Genera estadísticas del backtest"""
        df_trades = pd.DataFrame([asdict(t) for t in self.trades])
        
        total_trades = len(df_trades)
        executed_trades = df_trades['executed'].sum()
        failed_trades = total_trades - executed_trades
        
        total_pnl = df_trades['pnl'].sum()
        avg_pnl_per_trade = df_trades['pnl'].mean()
        
        winning_trades = df_trades[df_trades['pnl'] > 0]
        losing_trades = df_trades[df_trades['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        returns = np.diff(self.equity_curve) / self.equity_curve[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252 * 86400) if len(returns) > 0 else 0
        
        max_drawdown = self._calculate_max_drawdown(self.equity_curve)
        
        stats = {
            'total_trades': total_trades,
            'executed_trades': executed_trades,
            'failed_trades': failed_trades,
            'execution_rate': executed_trades / total_trades if total_trades > 0 else 0,
            'total_pnl': total_pnl,
            'total_return_pct': (self.capital - 100000) / 100000 * 100,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 else np.inf,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'avg_p_inclusion': df_trades['p_inclusion'].mean(),
            'avg_net_spread_bps': df_trades['net_spread'].mean() * 10000,
        }
        
        return stats
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calcula máximo drawdown"""
        peak = equity_curve[0]
        max_dd = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        return max_dd
    
    def plot_results(self):
        """Genera gráficos para el paper"""
        df_trades = pd.DataFrame([asdict(t) for t in self.trades])
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Equity curve
        axes[0, 0].plot(self.equity_curve)
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Trade #')
        axes[0, 0].set_ylabel('Capital ($)')
        axes[0, 0].grid(True)
        
        # 2. Distribución de spreads
        axes[0, 1].hist(df_trades['net_spread'] * 10000, bins=50, alpha=0.7)
        axes[0, 1].set_title('Net Spread Distribution')
        axes[0, 1].set_xlabel('Spread (bps)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].axvline(x=0, color='r', linestyle='--')
        axes[0, 1].grid(True)
        
        # 3. P_inclusión vs PnL
        executed = df_trades[df_trades['executed']]
        axes[1, 0].scatter(executed['p_inclusion'], executed['pnl'], alpha=0.5, s=10)
        axes[1, 0].set_title('P_Inclusión vs PnL')
        axes[1, 0].set_xlabel('P_Inclusión')
        axes[1, 0].set_ylabel('PnL ($)')
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].grid(True)
        
        # 4. Cumulative PnL
        axes[1, 1].plot(df_trades['pnl'].cumsum())
        axes[1, 1].set_title('Cumulative PnL')
        axes[1, 1].set_xlabel('Trade #')
        axes[1, 1].set_ylabel('Cumulative PnL ($)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=300)
        print("Gráficos guardados en 'backtest_results.png'")
        
    def export_trades(self, filename: str = 'trades.csv'):
        """Exporta trades para análisis"""
        df_trades = pd.DataFrame([asdict(t) for t in self.trades])
        df_trades.to_csv(filename, index=False)
        print(f"Trades exportados a '{filename}'")


# ============================================================================
# MAIN: Ejecución de simulación
# ============================================================================

def main():
    """Ejecuta simulación completa para paper académico"""
    
    print("=" * 70)
    print("SIMULACIÓN DE ARBITRAJE CEX-L2 CON MODELADO ESTOCÁSTICO")
    print("Paper Académico - Backtest con Datos Sintéticos")
    print("=" * 70)
    print()
    
    # 1. Genera datos sintéticos
    print("[1/4] Generando datos sintéticos...")
    simulator = DataSimulator(days=7, ticks_per_day=86400)  # 7 días, 1 tick/segundo
    data = simulator.generate_correlated_prices()
    print(f"  ✓ Generados {len(data)} ticks ({data['timestamp'].min()} a {data['timestamp'].max()})")
    print()
    
    # 2. Ejecuta backtest
    print("[2/4] Ejecutando backtest...")
    engine = BacktestEngine(
        data=data,
        min_spread=0.002,  # 0.2% = 20 bps
        min_p_inclusion=0.7,
        trade_size=1.0,
        initial_capital=100000
    )
    stats = engine.run_backtest()
    print()
    
    # 3. Muestra resultados
    print("[3/4] Resultados del Backtest:")
    print("=" * 70)
    print(f"Total Trades:           {stats['total_trades']}")
    print(f"Executed Trades:        {stats['executed_trades']} ({stats['execution_rate']:.1%})")
    print(f"Failed Trades:          {stats['failed_trades']}")
    print(f"-" * 70)
    print(f"Total PnL:              ${stats['total_pnl']:,.2f}")
    print(f"Total Return:           {stats['total_return_pct']:.2f}%")
    print(f"Avg PnL per Trade:      ${stats['avg_pnl_per_trade']:.2f}")
    print(f"-" * 70)
    print(f"Win Rate:               {stats['win_rate']:.1%}")
    print(f"Avg Win:                ${stats['avg_win']:.2f}")
    print(f"Avg Loss:               ${stats['avg_loss']:.2f}")
    print(f"Profit Factor:          {stats['profit_factor']:.2f}")
    print(f"-" * 70)
    print(f"Sharpe Ratio:           {stats['sharpe_ratio']:.2f}")
    print(f"Max Drawdown:           {stats['max_drawdown_pct']:.2f}%")
    print(f"-" * 70)
    print(f"Avg P_Inclusión:        {stats['avg_p_inclusion']:.1%}")
    print(f"Avg Net Spread:         {stats['avg_net_spread_bps']:.1f} bps")
    print("=" * 70)
    print()
    
    # 4. Genera visualizaciones
    print("[4/4] Generando gráficos...")
    engine.plot_results()
    engine.export_trades()
    print()
    print("✓ Simulación completada exitosamente")


if __name__ == "__main__":
    main()
