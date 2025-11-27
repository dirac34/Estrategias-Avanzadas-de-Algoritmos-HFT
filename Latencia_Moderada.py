"""
Sistema de Arbitraje CEX-L2 con Modelado Estocástico
Arquitectura asíncrona compatible con NautilusTrader
"""

import asyncio
import ccxt.pro as ccxtpro
from web3 import AsyncWeb3
from web3.middleware import geth_poa_middleware
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# (1) MÓDULO DE ADQUISICIÓN DE DATOS
# ============================================================================

@dataclass
class PriceFeed:
    """Datos de precio de CEX"""
    timestamp: datetime
    exchange: str
    symbol: str
    bid: float
    ask: float
    bid_volume: float
    ask_volume: float


@dataclass
class L2Transaction:
    """Transacción pendiente en L2 (mempool simulado)"""
    tx_hash: str
    timestamp: datetime
    token_in: str
    token_out: str
    amount_in: float
    priority_fee: float
    gas_limit: int


class CEXDataAcquisition:
    """Adquisición asíncrona de datos de CEX usando CCXT Pro"""
    
    def __init__(self, exchanges: List[str], symbols: List[str]):
        self.exchanges = {name: getattr(ccxtpro, name)() for name in exchanges}
        self.symbols = symbols
        self.price_feeds: Dict[str, PriceFeed] = {}
        self.running = False
        
    async def start(self):
        """Inicia la suscripción a orderbooks"""
        self.running = True
        tasks = []
        for exchange_name, exchange in self.exchanges.items():
            for symbol in self.symbols:
                tasks.append(self._watch_orderbook(exchange_name, exchange, symbol))
        await asyncio.gather(*tasks)
    
    async def _watch_orderbook(self, exchange_name: str, exchange, symbol: str):
        """Monitorea orderbook de un par en un exchange"""
        while self.running:
            try:
                orderbook = await exchange.watch_order_book(symbol)
                
                if orderbook['bids'] and orderbook['asks']:
                    feed = PriceFeed(
                        timestamp=datetime.utcnow(),
                        exchange=exchange_name,
                        symbol=symbol,
                        bid=orderbook['bids'][0][0],
                        ask=orderbook['asks'][0][0],
                        bid_volume=orderbook['bids'][0][1],
                        ask_volume=orderbook['asks'][0][1]
                    )
                    key = f"{exchange_name}:{symbol}"
                    self.price_feeds[key] = feed
                    
            except Exception as e:
                logger.error(f"Error watching {exchange_name} {symbol}: {e}")
                await asyncio.sleep(1)
    
    async def stop(self):
        """Detiene la adquisición"""
        self.running = False
        for exchange in self.exchanges.values():
            await exchange.close()


class L2MempoolMonitor:
    """Monitor simulado de transacciones pendientes en L2"""
    
    def __init__(self, rpc_url: str):
        self.w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        self.pending_txs: deque = deque(maxlen=1000)
        self.running = False
        
    async def start(self):
        """Inicia el monitoreo del mempool (simulado)"""
        self.running = True
        asyncio.create_task(self._simulate_mempool())
    
    async def _simulate_mempool(self):
        """Simula llegada de transacciones al mempool"""
        while self.running:
            # Simulación: genera transacciones con distribución Poisson
            await asyncio.sleep(np.random.exponential(0.5))  # Lambda = 2 tx/s
            
            tx = L2Transaction(
                tx_hash=f"0x{np.random.randint(0, 2**256):064x}",
                timestamp=datetime.utcnow(),
                token_in="USDC",
                token_out="ETH",
                amount_in=np.random.uniform(100, 10000),
                priority_fee=np.random.uniform(0.001, 0.01),
                gas_limit=np.random.randint(200000, 500000)
            )
            self.pending_txs.append(tx)
    
    def get_recent_txs(self, window_seconds: int = 60) -> List[L2Transaction]:
        """Obtiene transacciones recientes en ventana de tiempo"""
        cutoff = datetime.utcnow() - timedelta(seconds=window_seconds)
        return [tx for tx in self.pending_txs if tx.timestamp >= cutoff]
    
    async def stop(self):
        self.running = False


# ============================================================================
# (2) MÓDULO DE MODELADO ESTOCÁSTICO (Sequencer Modeler)
# ============================================================================

class ACDModel:
    """Autoregressive Conditional Duration (ACD) para modelar llegada de transacciones"""
    
    def __init__(self, omega: float = 0.1, alpha: float = 0.3, beta: float = 0.6):
        self.omega = omega  # Constante
        self.alpha = alpha  # Coef. de duración pasada
        self.beta = beta    # Coef. de duración condicional
        self.psi = 1.0      # Duración condicional esperada
        self.last_duration = 1.0
        
    def update(self, duration: float):
        """Actualiza el modelo con nueva duración observada"""
        self.psi = self.omega + self.alpha * self.last_duration + self.beta * self.psi
        self.last_duration = duration
        
    def get_expected_rate(self) -> float:
        """Retorna tasa de llegada esperada (lambda = 1/psi)"""
        return 1.0 / max(self.psi, 0.01)


class SequencerModeler:
    """Modela el comportamiento del sequencer L2 y calcula P_inclusión"""
    
    def __init__(self):
        self.acd_model = ACDModel()
        self.tx_history: deque = deque(maxlen=100)
        
    def calibrate(self, transactions: List[L2Transaction]):
        """Calibra el modelo ACD con transacciones observadas"""
        if len(transactions) < 2:
            return
            
        # Calcula duraciones entre transacciones
        sorted_txs = sorted(transactions, key=lambda x: x.timestamp)
        for i in range(1, len(sorted_txs)):
            duration = (sorted_txs[i].timestamp - sorted_txs[i-1].timestamp).total_seconds()
            self.acd_model.update(duration)
            self.tx_history.append(sorted_txs[i])
    
    def calculate_inclusion_probability(
        self, 
        our_priority_fee: float, 
        target_block_time: float = 2.0
    ) -> float:
        """
        Calcula P_inclusión basada en:
        - Tasa de llegada calibrada (lambda)
        - Priority fee relativo
        - Congestión actual
        """
        lambda_rate = self.acd_model.get_expected_rate()
        
        # Estima número de transacciones competidoras en próximo bloque
        expected_txs_in_block = lambda_rate * target_block_time
        
        # Calcula percentil de nuestro priority fee
        if len(self.tx_history) == 0:
            fee_percentile = 0.5
        else:
            recent_fees = [tx.priority_fee for tx in self.tx_history]
            fee_percentile = np.mean([our_priority_fee >= fee for fee in recent_fees])
        
        # Modelo simplificado: P_inclusión = f(percentil, congestión)
        congestion_factor = min(expected_txs_in_block / 100, 1.0)  # Block capacity = 100 txs
        
        p_inclusion = fee_percentile * (1 - 0.5 * congestion_factor)
        return np.clip(p_inclusion, 0.0, 1.0)


# ============================================================================
# (3) MÓDULO DE LÓGICA DE ARBITRAJE
# ============================================================================

@dataclass
class ArbitrageSignal:
    """Señal de arbitraje"""
    timestamp: datetime
    direction: str  # "CEX_TO_L2" o "L2_TO_CEX"
    cex_exchange: str
    symbol: str
    cex_price: float
    l2_price: float
    gross_spread: float
    net_spread: float
    size: float
    p_inclusion: float
    expected_profit: float


class UniswapV3Pricer:
    """Calcula precio y slippage en Uniswap V3"""
    
    def __init__(self, liquidity: float = 1000000, fee_tier: float = 0.003):
        self.liquidity = liquidity
        self.fee_tier = fee_tier
        
    def get_price_impact(self, amount: float, current_price: float) -> Tuple[float, float]:
        """
        Calcula precio de ejecución y slippage
        Modelo simplificado de liquidez constante
        """
        # Slippage aprox: sqrt(1 + 2*amount/liquidity) - 1
        slippage = np.sqrt(1 + 2 * amount / self.liquidity) - 1
        execution_price = current_price * (1 + slippage)
        total_cost = execution_price * (1 + self.fee_tier)
        
        return total_cost, slippage


class ArbitrageLógic:
    """Motor de detección y filtrado de oportunidades de arbitraje"""
    
    def __init__(
        self, 
        min_spread: float = 0.002,
        min_p_inclusion: float = 0.7,
        priority_fee_budget: float = 0.01
    ):
        self.min_spread = min_spread
        self.min_p_inclusion = min_p_inclusion
        self.priority_fee_budget = priority_fee_budget
        self.uniswap_pricer = UniswapV3Pricer()
        
    def evaluate_opportunity(
        self,
        cex_feed: PriceFeed,
        l2_price: float,
        p_inclusion: float,
        size: float
    ) -> Optional[ArbitrageSignal]:
        """
        Evalúa oportunidad de arbitraje y retorna señal si es viable
        """
        # Determina dirección
        if cex_feed.bid > l2_price:
            direction = "L2_TO_CEX"
            gross_spread = (cex_feed.bid - l2_price) / l2_price
            
            # Calcula costos L2
            l2_exec_price, slippage = self.uniswap_pricer.get_price_impact(size, l2_price)
            net_price_l2 = l2_exec_price
            net_price_cex = cex_feed.bid
            
        elif cex_feed.ask < l2_price:
            direction = "CEX_TO_L2"
            gross_spread = (l2_price - cex_feed.ask) / cex_feed.ask
            
            # Calcula costos
            l2_exec_price, slippage = self.uniswap_pricer.get_price_impact(size, l2_price)
            net_price_l2 = l2_exec_price
            net_price_cex = cex_feed.ask
        else:
            return None
        
        # Descuenta priority fee y costos de transacción
        priority_fee_cost = self.priority_fee_budget / size  # Costo por unidad
        net_spread = gross_spread - slippage - priority_fee_cost - 0.001  # 0.1% taker fee CEX
        
        # Filtra por umbral de spread y P_inclusión
        if net_spread < self.min_spread:
            return None
        
        if p_inclusion < self.min_p_inclusion:
            logger.info(f"P_inclusión muy baja: {p_inclusion:.2%} < {self.min_p_inclusion:.2%}")
            return None
        
        expected_profit = net_spread * size * net_price_cex
        
        return ArbitrageSignal(
            timestamp=datetime.utcnow(),
            direction=direction,
            cex_exchange=cex_feed.exchange,
            symbol=cex_feed.symbol,
            cex_price=net_price_cex,
            l2_price=net_price_l2,
            gross_spread=gross_spread,
            net_spread=net_spread,
            size=size,
            p_inclusion=p_inclusion,
            expected_profit=expected_profit
        )


# ============================================================================
# (4) MÓDULO DE EJECUCIÓN ASÍNCRONA
# ============================================================================

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Resultado de ejecución"""
    success: bool
    order_id: Optional[str]
    tx_hash: Optional[str]
    filled_price: Optional[float]
    filled_size: Optional[float]
    error: Optional[str]


class CEXExecutor:
    """Ejecutor de órdenes en CEX"""
    
    def __init__(self, exchange_name: str, api_key: str, secret: str):
        self.exchange = getattr(ccxtpro, exchange_name)({
            'apiKey': api_key,
            'secret': secret,
            'enableRateLimit': True
        })
        
    async def execute_order(
        self, 
        symbol: str, 
        side: str, 
        size: float, 
        order_type: str = 'market'
    ) -> ExecutionResult:
        """Ejecuta orden en CEX"""
        try:
            order = await self.exchange.create_order(
                symbol=symbol,
                type=order_type,
                side=side,
                amount=size
            )
            
            return ExecutionResult(
                success=True,
                order_id=order['id'],
                tx_hash=None,
                filled_price=order.get('price'),
                filled_size=order.get('filled'),
                error=None
            )
        except Exception as e:
            logger.error(f"Error ejecutando orden CEX: {e}")
            return ExecutionResult(
                success=False,
                order_id=None,
                tx_hash=None,
                filled_price=None,
                filled_size=None,
                error=str(e)
            )


class L2Executor:
    """Ejecutor de transacciones en L2 (wrapper de web3.py)"""
    
    def __init__(self, rpc_url: str, private_key: str, router_address: str):
        self.w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(rpc_url))
        self.account = self.w3.eth.account.from_key(private_key)
        self.router_address = router_address
        
    async def execute_swap(
        self,
        token_in: str,
        token_out: str,
        amount_in: float,
        min_amount_out: float,
        priority_fee: float
    ) -> ExecutionResult:
        """Ejecuta swap en L2 (Uniswap V3 / similares)"""
        try:
            # Construye transacción (simulado)
            nonce = await self.w3.eth.get_transaction_count(self.account.address)
            
            tx = {
                'from': self.account.address,
                'to': self.router_address,
                'value': 0,
                'gas': 300000,
                'maxFeePerGas': self.w3.to_wei(50, 'gwei'),
                'maxPriorityFeePerGas': self.w3.to_wei(priority_fee, 'gwei'),
                'nonce': nonce,
                'chainId': await self.w3.eth.chain_id,
                # data: encodeFunctionCall('swap', [...]) - simplificado
            }
            
            signed_tx = self.account.sign_transaction(tx)
            tx_hash = await self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Espera confirmación (con timeout)
            receipt = await asyncio.wait_for(
                self.w3.eth.wait_for_transaction_receipt(tx_hash),
                timeout=30
            )
            
            return ExecutionResult(
                success=receipt['status'] == 1,
                order_id=None,
                tx_hash=tx_hash.hex(),
                filled_price=None,  # Se obtendría del evento del contrato
                filled_size=amount_in,
                error=None if receipt['status'] == 1 else "Transaction reverted"
            )
            
        except Exception as e:
            logger.error(f"Error ejecutando swap L2: {e}")
            return ExecutionResult(
                success=False,
                order_id=None,
                tx_hash=None,
                filled_price=None,
                filled_size=None,
                error=str(e)
            )


class AsyncExecutionEngine:
    """Motor de ejecución concurrente de CEX + L2"""
    
    def __init__(self, cex_executor: CEXExecutor, l2_executor: L2Executor):
        self.cex_executor = cex_executor
        self.l2_executor = l2_executor
        
    async def execute_arbitrage(
        self, 
        signal: ArbitrageSignal, 
        priority_fee: float
    ) -> Tuple[ExecutionResult, ExecutionResult]:
        """
        Ejecuta ambas patas del arbitraje concurrentemente
        """
        if signal.direction == "L2_TO_CEX":
            # Compra en L2, vende en CEX
            l2_task = self.l2_executor.execute_swap(
                token_in="USDC",
                token_out="ETH",
                amount_in=signal.size,
                min_amount_out=signal.size / signal.l2_price * 0.99,
                priority_fee=priority_fee
            )
            cex_task = self.cex_executor.execute_order(
                symbol=signal.symbol,
                side='sell',
                size=signal.size
            )
        else:
            # Compra en CEX, vende en L2
            cex_task = self.cex_executor.execute_order(
                symbol=signal.symbol,
                side='buy',
                size=signal.size
            )
            l2_task = self.l2_executor.execute_swap(
                token_in="ETH",
                token_out="USDC",
                amount_in=signal.size,
                min_amount_out=signal.size * signal.l2_price * 0.99,
                priority_fee=priority_fee
            )
        
        # Ejecuta concurrentemente
        results = await asyncio.gather(l2_task, cex_task, return_exceptions=True)
        return results[0], results[1]


# ============================================================================
# (5) MÓDULO DE GESTIÓN DE RIESGOS
# ============================================================================

class RiskManager:
    """Gestión de riesgos con Volatility Targeting y Failover"""
    
    def __init__(
        self,
        max_position_size: float = 10.0,
        target_volatility: float = 0.15,
        max_drawdown: float = 0.05,
        failover_threshold: float = 0.03
    ):
        self.max_position_size = max_position_size
        self.target_volatility = target_volatility
        self.max_drawdown = max_drawdown
        self.failover_threshold = failover_threshold
        
        self.equity_curve: deque = deque(maxlen=100)
        self.returns: deque = deque(maxlen=100)
        self.peak_equity = 100000.0
        self.current_equity = 100000.0
        
    def calculate_position_size(self, signal: ArbitrageSignal) -> float:
        """
        Ajusta tamaño de posición basado en Volatility Targeting
        """
        if len(self.returns) < 20:
            realized_vol = self.target_volatility
        else:
            realized_vol = np.std(list(self.returns)) * np.sqrt(252)
        
        # Ajuste de tamaño: position_size = target_vol / realized_vol * base_size
        vol_scalar = self.target_volatility / max(realized_vol, 0.01)
        adjusted_size = signal.size * vol_scalar
        
        return min(adjusted_size, self.max_position_size)
    
    def update_equity(self, pnl: float):
        """Actualiza curva de equity"""
        self.current_equity += pnl
        self.equity_curve.append(self.current_equity)
        
        if len(self.equity_curve) > 1:
            ret = pnl / self.equity_curve[-2]
            self.returns.append(ret)
        
        self.peak_equity = max(self.peak_equity, self.current_equity)
    
    def check_failover(self) -> bool:
        """
        Verifica si se debe activar protocolo de failover
        Retorna True si se debe cerrar todas las posiciones
        """
        # Calcula drawdown actual
        drawdown = (self.peak_equity - self.current_equity) / self.peak_equity
        
        if drawdown >= self.max_drawdown:
            logger.error(f"MAX DRAWDOWN ALCANZADO: {drawdown:.2%}")
            return True
        
        # Verifica pérdida reciente
        if len(self.returns) >= 10:
            recent_return = sum(list(self.returns)[-10:])
            if recent_return <= -self.failover_threshold:
                logger.error(f"PÉRDIDA RECIENTE EXCESIVA: {recent_return:.2%}")
                return True
        
        return False
    
    async def emergency_close(self, cex_executor: CEXExecutor):
        """Cierre de emergencia en CEX"""
        logger.critical("EJECUTANDO CIERRE DE EMERGENCIA")
        # Implementar lógica de cierre de todas las posiciones
        # ...


# ============================================================================
# (6) ARQUITECTURA PRINCIPAL (Main Loop)
# ============================================================================

class ArbitrageBot:
    """Sistema principal de arbitraje CEX-L2"""
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Inicializa componentes
        self.cex_data = CEXDataAcquisition(
            exchanges=config['cex_exchanges'],
            symbols=config['symbols']
        )
        self.l2_monitor = L2MempoolMonitor(config['l2_rpc_url'])
        self.sequencer_modeler = SequencerModeler()
        self.arbitrage_logic = ArbitrageLógic(
            min_spread=config['min_spread'],
            min_p_inclusion=config['min_p_inclusion']
        )
        
        self.cex_executor = CEXExecutor(
            exchange_name=config['cex_exchanges'][0],
            api_key=config['cex_api_key'],
            secret=config['cex_secret']
        )
        self.l2_executor = L2Executor(
            rpc_url=config['l2_rpc_url'],
            private_key=config['l2_private_key'],
            router_address=config['uniswap_router']
        )
        self.execution_engine = AsyncExecutionEngine(
            self.cex_executor,
            self.l2_executor
        )
        
        self.risk_manager = RiskManager(
            max_position_size=config['max_position_size'],
            target_volatility=config['target_volatility']
        )
        
        self.running = False
        
    async def start(self):
        """Inicia el bot"""
        logger.info("Iniciando ArbitrageBot...")
        self.running = True
        
        # Inicia adquisición de datos
        await asyncio.gather(
            self.cex_data.start(),
            self.l2_monitor.start()
        )
        
        # Inicia main loop
        await self.main_loop()
    
    async def main_loop(self):
        """Loop principal de operación"""
        while self.running:
            try:
                # 1. Calibra modelo de sequencer con transacciones recientes
                recent_txs = self.l2_monitor.get_recent_txs(window_seconds=60)
                self.sequencer_modeler.calibrate(recent_txs)
                
                # 2. Calcula P_inclusión para diferentes priority fees
                priority_fee = 2.0  # gwei
                p_inclusion = self.sequencer_modeler.calculate_inclusion_probability(
                    our_priority_fee=priority_fee
                )
                
                # 3. Evalúa oportunidades de arbitraje
                for key, cex_feed in self.cex_data.price_feeds.items():
                    # Simula precio L2 (en producción se obtendría de Uniswap)
                    l2_price = cex_feed.bid * (1 + np.random.uniform(-0.005, 0.005))
                    
                    signal = self.arbitrage_logic.evaluate_opportunity(
                        cex_feed=cex_feed,
                        l2_price=l2_price,
                        p_inclusion=p_inclusion,
                        size=1.0  # ETH
                    )
                    
                    if signal:
                        logger.info(f"SEÑAL DETECTADA: {signal.direction} | "
                                  f"Spread: {signal.net_spread:.2%} | "
                                  f"P_inc: {signal.p_inclusion:.2%} | "
                                  f"Profit: ${signal.expected_profit:.2f}")
                        
                        # 4. Ajusta tamaño de posición con risk management
                        adjusted_size = self.risk_manager.calculate_position_size(signal)
                        signal.size = adjusted_size
                        
                        # 5. Verifica failover
                        if self.risk_manager.check_failover():
                            await self.risk_manager.emergency_close(self.cex_executor)
                            self.running = False
                            break
                        
                        # 6. Ejecuta arbitraje
                        l2_result, cex_result = await self.execution_engine.execute_arbitrage(
                            signal, priority_fee
                        )
                        
                        # 7. Actualiza risk management
                        if l2_result.success and cex_result.success:
                            pnl = signal.expected_profit
                            self.risk_manager.update_equity(pnl)
                            logger.info(f"ARBITRAJE EXITOSO | PnL: ${pnl:.2f}")
                        else:
                            logger.error(f"ARBITRAJE FALLIDO | L2: {l2_result.error} | CEX: {cex_result.error}")
                            self.risk_manager.update_equity(-signal.size * 0.01)  # Pérdida estimada
                
                # Espera antes de siguiente iteración
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error en main loop: {e}")
                await asyncio.sleep(1)
    
    async def stop(self):
        """Detiene el bot"""
        logger.info("Deteniendo ArbitrageBot...")
        self.running = False
        await self.cex_data.stop()
        await self.l2_monitor.stop()


# ============================================================================
# PUNTO DE ENTRADA
# ============================================================================

async def main():
    """Función principal"""
    
    config = {
        'cex_exchanges': ['binance', 'kraken'],
        'symbols': ['ETH/USDT'],
        'l2_rpc_url': 'https://mainnet.optimism.io',
        'cex_api_key': 'YOUR_API_KEY',
        'cex_secret': 'YOUR_SECRET',
        'l2_private_key': 'YOUR_PRIVATE_KEY',
        'uniswap_router': '0xE592427A0AEce92De3Edee1F18E0157C05861564',
        'min_spread': 0.002,  # 0.2%
        'min_p_inclusion': 0.7,
        'max_position_size': 5.0,  # ETH
        'target_volatility': 0.15
    }
    
    bot = ArbitrageBot(config)
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        logger.info("Interrupción por usuario")
    finally:
        await bot.stop()


if __name__ == "__main__":
    asyncio.run(main())
