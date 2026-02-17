import numpy as np
from datetime import datetime, timedelta


# ============================================================
# BASE CLASS (NO AUTO REFILL LOGIC)
# ============================================================
class BaseStationSimulator:

    def __init__(
        self,
        station_id,
        tank_capacity=6000,
    ):
        self.station_id = station_id
        self.tank_capacity = tank_capacity

        self.current_stock = tank_capacity
        self.current_time = datetime.now().replace(second=0, microsecond=0)

    # ------------------------------------------
    # Must be overridden
    # ------------------------------------------
    def hourly_profile(self, hour):
        raise NotImplementedError

    def generate_flow(self):
        raise NotImplementedError

    # ------------------------------------------
    # SCADA EVENT (NO AUTO REFILL)
    # ------------------------------------------
    def next_event(self):

        self.current_time += timedelta(minutes=1)

        flow = self.generate_flow()

        # Reduce stock
        self.current_stock -= flow
        self.current_stock = max(self.current_stock, 0)

        return {
            "timestamp": self.current_time,
            "station_id": self.station_id,
            "flow_rate": round(flow, 3),
            "current_stock": round(self.current_stock, 2)
        }

    # ------------------------------------------
    # Manual refill (Triggered externally)
    # ------------------------------------------
    def refill(self, amount):
        self.current_stock += amount
        self.current_stock = min(self.current_stock, self.tank_capacity)

        return {
            "station_id": self.station_id,
            "new_stock": round(self.current_stock, 2)
        }

    # ------------------------------------------
    # Hard reset (Sensor correction / override)
    # ------------------------------------------
    def reset_stock(self, new_amount):
        self.current_stock = min(new_amount, self.tank_capacity)

        return {
            "station_id": self.station_id,
            "reset_stock": round(self.current_stock, 2)
        }


# ============================================================
# STATION A — High Volume Pattern
# ============================================================
class StationA(BaseStationSimulator):

    def __init__(self, station_id, mean_daily, std_daily, weekend_multiplier):
        super().__init__(station_id)
        self.mean_daily = mean_daily
        self.std_daily = std_daily
        self.weekend_multiplier = weekend_multiplier

        # Convert daily demand into per-minute baseline
        self.base_daily_demand = mean_daily

        # Hourly profile derived from your generator weights
        self.hourly_weights = np.array([
            0.03 if 0 <= h <= 5 else
            0.06 if 6 <= h <= 10 else
            0.045 if 11 <= h <= 16 else
            0.07 if 17 <= h <= 22 else
            0.04
            for h in range(24)
        ])

        self.hourly_weights = self.hourly_weights / self.hourly_weights.sum()

    # --------------------------------------------------
    # HOURLY PROFILE (returns kg/min baseline)
    # --------------------------------------------------
    def hourly_profile(self, hour, weekend=False):

        daily_demand = np.random.normal(
            self.base_daily_demand,
            self.std_daily * 0.25
        )

        if weekend:
            daily_demand *= self.weekend_multiplier

        daily_demand = max(daily_demand, self.mean_daily * 0.6)

        hourly_demand = daily_demand * self.hourly_weights[hour]

        return hourly_demand / 60.0  # per minute base flow

    # --------------------------------------------------
    # FLOW GENERATOR
    # --------------------------------------------------
    def generate_flow(self):

        hour = self.current_time.hour
        minute = self.current_time.minute
        dow = self.current_time.weekday()
        weekend = dow >= 5

        base = self.hourly_profile(hour, weekend)

        # Smooth intra-hour variation
        minute_wave = 0.08 * base * np.sin(2 * np.pi * minute / 60)

        # Gaussian noise
        noise = np.random.normal(0, base * 0.12)

        # Rare heavy-vehicle spike
        if np.random.rand() < 0.002:
            noise += np.random.uniform(1.5, 3.5)

        flow = base + minute_wave + noise

        return max(0.1, flow)



# ============================================================
# STATION B — Balanced Demand Pattern
# ============================================================
class Station_A(BaseStationSimulator):

    def hourly_profile(self, hour):

        if 0 <= hour <= 4:
            return 0.5
        elif 5 <= hour <= 8:
            return 1.4
        elif 9 <= hour <= 15:
            return 1.0
        elif 16 <= hour <= 20:
            return 1.7
        elif 21 <= hour <= 23:
            return 0.8

    def generate_flow(self):

        hour = self.current_time.hour
        minute = self.current_time.minute
        weekend = self.current_time.weekday() >= 5

        base = self.hourly_profile(hour)

        if weekend:
            base *= 0.92

        minute_wave = 0.07 * base * np.sin(2 * np.pi * minute / 60)
        noise = np.random.normal(0, base * 0.10)

        # rare spike event
        if np.random.rand() < 0.0015:
            noise += np.random.uniform(base * 0.8, base * 1.5)

        flow = base + minute_wave + noise

        return max(0.3, round(flow, 4))


# ============================================================
# STATION C — Low Volume Pattern
# ============================================================
class Station_C(BaseStationSimulator):
    def hourly_profile(self, hour):

        # Based on generator hourly_weight structure

        if 0 <= hour <= 5:
            return 0.6
        elif 6 <= hour <= 9:
            return 1.5
        elif 10 <= hour <= 15:
            return 1.1
        elif 16 <= hour <= 19:
            return 1.8
        elif 20 <= hour <= 22:
            return 1.3
        else:
            return 0.8

    def generate_flow(self):

        hour = self.current_time.hour
        minute = self.current_time.minute
        weekend = self.current_time.weekday() >= 5

        base = self.hourly_profile(hour)

        # Weekend adjustment (mild amplification)
        if weekend:
            base *= 1.05

        # Stronger intra-hour oscillation (traffic bunching)
        minute_wave = 0.12 * base * np.sin(2 * np.pi * minute / 60)

        # Higher stochastic variation
        noise = np.random.normal(0, base * 0.15)

        # More frequent rare spikes
        if np.random.rand() < 0.003:
            noise += np.random.uniform(base * 0.7, base * 2.0)

        flow = base + minute_wave + noise

        return max(0.4, round(flow, 4))


# ============================================================
# SCADA MANAGER
# ============================================================
class SCADAMultiStation:

    def __init__(self):
        self.stations = {}

    def add_station(self, station_obj):
        self.stations[station_obj.station_id] = station_obj

    # Generate next event for one station
    def next_event(self, station_id):
        return self.stations[station_id].next_event()

    # Generate next event for all stations
    def next_all(self):
        return [
            station.next_event()
            for station in self.stations.values()
        ]

    # Manual refill
    def refill_station(self, station_id, amount):
        return self.stations[station_id].refill(amount)

    # Hard reset
    def reset_station(self, station_id, new_amount):
        return self.stations[station_id].reset_stock(new_amount)